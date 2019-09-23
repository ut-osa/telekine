#include "../include/hip/hcc_detail/code_object_bundle.hpp"
#include "hip_cpp_bridge.h"

#include "hip_hcc_internal.h"
//#include "hsa_helpers.hpp"
#include "trace_helper.h"

#include "elfio/elfio.hpp"

#include <link.h>
#include <hsa_limited.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace ELFIO;
using namespace hip_impl;
using namespace std;

#define hsa_executable_symbol_get_info(x, y, z) ({\
   auto __tmp = z; \
   auto ret = __do_c_hsa_executable_symbol_get_info(x, y, (char *)z, sizeof(*__tmp)); \
   ret;\
})

inline constexpr bool operator==(hsa_isa_t x, hsa_isa_t y) { return x.handle == y.handle; }

namespace std {
template <>
struct hash<hsa_isa_t> {
    size_t operator()(hsa_isa_t x) const { return hash<decltype(x.handle)>{}(x.handle); }
};
}  // namespace std

namespace hip_impl {

inline hipError_t get_mass_symbol_info(size_t n,
                                const hsa_executable_symbol_t *symbols,
                                hsa_symbol_kind_t *types,
                                hipFunction_t *descriptors,
                                hsa_agent_t *agents,
                                char **names, char *name_string_pool,
                                size_t pool_size)
{
   hipError_t r;
   vector<unsigned> offsets(n);
   r = __do_c_mass_symbol_info(n, symbols, types, descriptors, (uint8_t *)agents,
                               offsets.data(), name_string_pool, pool_size);

   if (r == hipSuccess) {
      for (unsigned i = 0; i < n; i++)
         names[i] = name_string_pool + offsets[i];
   }
   return r;
}

inline hsa_symbol_kind_t type(hsa_executable_symbol_t x) {
    hsa_symbol_kind_t r = {};
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &r);

    return r;
}

inline std::string name(hsa_executable_symbol_t x) {
    std::uint32_t sz = 0u;
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &sz);

    std::string r(sz, '\0');
   __do_c_hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_NAME,
                                         &r.front(), sz);
    return r;
}

inline hsa_agent_t agent(hsa_executable_symbol_t x) {
    hsa_agent_t r = {};
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_AGENT, &r);

    return r;
}

}  // namespace hip_impl

namespace {
struct Symbol {
    string name;
    ELFIO::Elf64_Addr value = 0;
    Elf_Xword size = 0;
    Elf_Half sect_idx = 0;
    uint8_t bind = 0;
    uint8_t type = 0;
    uint8_t other = 0;
};

inline Symbol read_symbol(const symbol_section_accessor& section, unsigned int idx) {
    assert(idx < section.get_symbols_num());

    Symbol r;
    section.get_symbol(idx, r.name, r.value, r.size, r.bind, r.type, r.sect_idx, r.other);

    return r;
}

template <typename P>
inline section* find_section_if(elfio& reader, P p) {
    const auto it = find_if(reader.sections.begin(), reader.sections.end(), move(p));

    return it != reader.sections.end() ? *it : nullptr;
}

const std::unordered_map<std::string, std::pair<ELFIO::Elf64_Addr, ELFIO::Elf_Xword>>&
symbol_addresses() {
    static unordered_map<string, pair<Elf64_Addr, Elf_Xword>> r;
    static once_flag f;

    call_once(f, []() {
        dl_iterate_phdr(
            [](dl_phdr_info* info, size_t, void*) {
                static constexpr const char self[] = "/proc/self/exe";
                elfio reader;

                static unsigned int iter = 0u;
                if (reader.load(!iter ? self : info->dlpi_name)) {
                    auto it = find_section_if(
                        reader, [](const class section* x) { return x->get_type() == SHT_SYMTAB; });

                    if (it) {
                        const symbol_section_accessor symtab{reader, it};

                        for (auto i = 0u; i != symtab.get_symbols_num(); ++i) {
                            auto tmp = read_symbol(symtab, i);

                            if (tmp.type == STT_OBJECT && tmp.sect_idx != SHN_UNDEF) {
                                const auto addr = tmp.value + (iter ? info->dlpi_addr : 0);
                                r.emplace(move(tmp.name), make_pair(addr, tmp.size));
                            }
                        }
                    }

                    ++iter;
                }

                return 0;
            },
            nullptr);
    });

    return r;
}

#if 0
void associate_code_object_symbols_with_host_allocation(const elfio& reader,
                                                        section* code_object_dynsym,
                                                        hsa_agent_t agent,
                                                        hsa_executable_t executable) {
    if (!code_object_dynsym) return;

    const auto undefined_symbols =
        copy_names_of_undefined_symbols(symbol_section_accessor{reader, code_object_dynsym});

    for (auto&& x : undefined_symbols) {
        if (globals().find(x) != globals().cend()) return;

        const auto it1 = symbol_addresses().find(x);

        if (it1 == symbol_addresses().cend()) {
            throw runtime_error{"Global symbol: " + x + " is undefined."};
        }

        static mutex mtx;
        lock_guard<mutex> lck{mtx};

        if (globals().find(x) != globals().cend()) return;
        globals().emplace(x, (void*)(it1->second.first));
        void* p = nullptr;
        hsa_amd_memory_lock(reinterpret_cast<void*>(it1->second.first), it1->second.second,
                            nullptr,  // All agents.
                            0, &p);

        hsa_executable_agent_global_variable_define(executable, agent, x.c_str(), p);
    }
}
#endif

vector<char> code_object_blob_for_process() {
    static constexpr const char self[] = "/proc/self/exe";
    static constexpr const char kernel_section[] = ".kernel";

    elfio reader;

    if (!reader.load(self)) {
        throw runtime_error{"Failed to load ELF file for current process."};
    }

    auto kernels =
        find_section_if(reader, [](const section* x) { return x->get_name() == kernel_section; });

    vector<char> r;
    if (kernels) {
        r.insert(r.end(), kernels->get_data(), kernels->get_data() + kernels->get_size());
    }

    return r;
}

const unordered_map<hsa_isa_t, vector<vector<char>>>& code_object_blobs() {
    static unordered_map<hsa_isa_t, vector<vector<char>>> r;
    static once_flag f;

    call_once(f, []() {
        static vector<vector<char>> blobs{code_object_blob_for_process()};

        dl_iterate_phdr(
            [](dl_phdr_info* info, std::size_t, void*) {
                elfio tmp;
                if (tmp.load(info->dlpi_name)) {
                    const auto it = find_section_if(
                        tmp, [](const section* x) { return x->get_name() == ".kernel"; });

                    if (it) blobs.emplace_back(it->get_data(), it->get_data() + it->get_size());
                }
                return 0;
            },
            nullptr);

        for (auto&& blob : blobs) {
            Bundled_code_header tmp{blob};
            if (valid(tmp)) {
                for (auto&& bundle : bundles(tmp)) {
                    r[triple_to_hsa_isa(bundle.triple)].push_back(bundle.blob);
                }
            }
        }
    });

    return r;
}

vector<pair<uintptr_t, string>> function_names_for(const elfio& reader, section* symtab) {
    vector<pair<uintptr_t, string>> r;
    symbol_section_accessor symbols{reader, symtab};

    for (auto i = 0u; i != symbols.get_symbols_num(); ++i) {
        // TODO: this is boyscout code, caching the temporaries
        //       may be of worth.
        auto tmp = read_symbol(symbols, i);

        if (tmp.type == STT_FUNC && tmp.sect_idx != SHN_UNDEF && !tmp.name.empty()) {
            r.emplace_back(tmp.value, tmp.name);
        }
    }

    return r;
}

const vector<pair<uintptr_t, string>>& function_names_for_process() {
    static constexpr const char self[] = "/proc/self/exe";

    static vector<pair<uintptr_t, string>> r;
    static once_flag f;

    call_once(f, []() {
        elfio reader;

        if (!reader.load(self)) {
            throw runtime_error{"Failed to load the ELF file for the current process."};
        }

        auto symtab =
            find_section_if(reader, [](const section* x) { return x->get_type() == SHT_SYMTAB; });

        if (symtab) r = function_names_for(reader, symtab);
    });

    return r;
}

#define MAX_SYMBOLS 4096
struct kernel_info {
   hsa_executable_symbol_t symbol;
   hipFunction_t descriptor;
   hsa_agent_t  agent;

   kernel_info(hsa_executable_symbol_t s, hipFunction_t f, hsa_agent_t a) :
      symbol(s), descriptor(f), agent(a) {}
};

const unordered_map<string, vector<kernel_info>>& kernels() {
    static unordered_map<string, vector<kernel_info>> r;
    static once_flag f;

    call_once(f, []() {
        for (auto&& agent_executables : executables()) {
            for (auto&& executable : agent_executables.second) {
               size_t n_symbols;
               hsa_executable_symbol_t symbols[MAX_SYMBOLS];

               n_symbols = __do_c_get_kerenel_symbols(&executable,
                                                      &agent_executables.first,
                                                      symbols, MAX_SYMBOLS);
               vector<char *> names(n_symbols);
               vector<hsa_symbol_kind_t> types(n_symbols);
               vector<char> name_string_pool(n_symbols * 256);
               vector<hipFunction_t> descriptors(n_symbols);
               vector<hsa_agent_t> agents(n_symbols);
               for (unsigned i = 0; i < n_symbols; i++)
                  agents[i].handle = -1;

               get_mass_symbol_info(n_symbols, symbols, types.data(),
                                    descriptors.data(), agents.data(),
                                    names.data(), name_string_pool.data(),
                                    name_string_pool.size());
               /*
               for (auto s = symbols; s < symbols + n_symbols; s++)
                  if (type(*s) == HSA_SYMBOL_KIND_KERNEL) r[name(*s)].push_back(*s);
                  */
               for (unsigned i = 0; i < n_symbols; i++)
                  if(types[i] == HSA_SYMBOL_KIND_KERNEL)
                     r[string(names[i])].push_back(kernel_info(symbols[i], descriptors[i], agents[i]));

            }
        }
    });

    return r;
}

}  // namespace

#define MAX_AGENTS 16
#define MAX_ISAS 16
namespace hip_impl {
Kernel_descriptor::Kernel_descriptor(std::uint64_t kernel_object, const std::string& name)
  : kernel_object_{kernel_object}, name_{name}
{
#if 0
  bool supported{false};
  std::uint16_t min_v{UINT16_MAX};
  auto r = nw_hsa_system_major_extension_supported(
      HSA_EXTENSION_AMD_LOADER, 1, &min_v, &supported);

  if (r != HSA_STATUS_SUCCESS || !supported) return;
#endif

  auto r = HSA_STATUS_SUCCESS;
  r = __do_c_query_host_address(kernel_object_, reinterpret_cast<char *>(&kernel_header_buffer));
#if 0
  hsa_ven_amd_loader_1_01_pfn_t tbl{};

  r = hsa_system_get_major_extension_table(
      HSA_EXTENSION_AMD_LOADER,
      1,
      sizeof(tbl),
      reinterpret_cast<void*>(&tbl));

  if (r != HSA_STATUS_SUCCESS) return;
  if (!tbl.hsa_ven_amd_loader_query_host_address) return;

  r = tbl.hsa_ven_amd_loader_query_host_address(
      reinterpret_cast<void*>(kernel_object_),
      reinterpret_cast<const void**>(&kernel_header_));
#endif
  if (r != HSA_STATUS_SUCCESS) return;
  kernel_header_ = &kernel_header_buffer;
}

const unordered_map<hsa_agent_t, vector<hsa_executable_t>>&
executables() {  // TODO: This leaks the hsa_executable_ts, it should use RAII.
    static unordered_map<hsa_agent_t, vector<hsa_executable_t>> r;
    static once_flag f;

    call_once(f, []() {
#if 0
        static const auto accelerators = hc::accelerator::get_all();

        for (auto&& acc : accelerators) {
            auto agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());
            if (!agent || !acc.is_hsa_accelerator()) continue;
#endif
        hsa_agent_t agents[MAX_AGENTS];
        size_t n_agents = __do_c_get_agents(agents, MAX_AGENTS);

        for (auto agent = agents; agent < agents + n_agents; ++agent) {
#if 0
            hsa_agent_iterate_isas(*agent,
                                   [](hsa_isa_t x, void* pa) {
#endif
            hsa_isa_t isas[MAX_ISAS];
            size_t n_isas = __do_c_get_isas(*agent, isas, MAX_ISAS);
            for (auto isa = isas; isa < isas + n_isas; isa++) {
               const auto it = code_object_blobs().find(*isa);

               if (it != code_object_blobs().cend()) {
                   for (auto&& blob : it->second) {
                       hsa_executable_t tmp = {};

                       nw_hsa_executable_create_alt(
                           HSA_PROFILE_FULL,
                           HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                           &tmp);

                       // TODO: this is massively inefficient and only
                       //       meant for illustration.
                       string blob_to_str{blob.cbegin(), blob.cend()};
                       tmp = load_executable(blob_to_str, tmp, *agent);

                       if (tmp.handle) r[*agent].push_back(tmp);
                   }
               }
           }
        }
    });

    return r;
}

const unordered_map<uintptr_t, string>& function_names() {
    static unordered_map<uintptr_t, string> r{function_names_for_process().cbegin(),
                                              function_names_for_process().cend()};
    static once_flag f;

    call_once(f, []() {
        dl_iterate_phdr(
            [](dl_phdr_info* info, size_t, void*) {
                elfio tmp;
                if (tmp.load(info->dlpi_name)) {
                    const auto it = find_section_if(
                        tmp, [](const section* x) { return x->get_type() == SHT_SYMTAB; });

                    if (it) {
                        auto n = function_names_for(tmp, it);

                        for (auto&& f : n) f.first += info->dlpi_addr;

                        r.insert(make_move_iterator(n.begin()), make_move_iterator(n.end()));
                    }
                }

                return 0;
            },
            nullptr);
    });

    return r;
}

const unordered_map<uintptr_t, vector<pair<hsa_agent_t, hipFunction_t>>>& functions() {
    static unordered_map<uintptr_t, vector<pair<hsa_agent_t, hipFunction_t>>> r;
    static once_flag f;

    call_once(f, []() {
        for (auto&& function : function_names()) {
            const auto it = kernels().find(function.second);

            if (it != kernels().cend()) {
                for (auto&& kernel_symbol : it->second) {
                /*
                    hipFunction_t func;
                    __do_c_get_kernel_descriptor(&kernel_symbol.symbol, it->first.c_str(),
                                                 &func);
                                                 */
                    r[function.first].emplace_back(
                        kernel_symbol.agent /*agent(kernel_symbol.symbol)*/,
                        kernel_symbol.descriptor/* func */);
#if 0
                        Kernel_descriptor{kernel_object(kernel_symbol), it->first});
#endif
                }
            }
        }
    });

    return r;
}

unordered_map<string, void*>& globals() {
    static unordered_map<string, void*> r;
    static once_flag f;
    call_once(f, []() { r.reserve(symbol_addresses().size()); });

    return r;
}

hsa_executable_t load_executable(const string& file, hsa_executable_t executable,
                                 hsa_agent_t agent) {
/*
    elfio reader;
    stringstream tmp{file};

    if (!reader.load(tmp)) return hsa_executable_t{};

    const auto code_object_dynsym = find_section_if(
        reader, [](const ELFIO::section* x) { return x->get_type() == SHT_DYNSYM; });

    associate_code_object_symbols_with_host_allocation(reader, code_object_dynsym, agent,
                                                       executable);

    load_code_object_and_freeze_executable(file, agent, executable);

    return executable;
*/
   __do_c_load_executable(file.data(), file.size(), &executable, &agent);
   return executable;
}

// To force HIP to load the kernels and to setup the function
// symbol map on program startup
/*
class startup_kernel_loader {
   private:
    startup_kernel_loader() { functions(); }
    startup_kernel_loader(const startup_kernel_loader&) = delete;
    startup_kernel_loader& operator=(const startup_kernel_loader&) = delete;
    static startup_kernel_loader skl;
};
startup_kernel_loader startup_kernel_loader::skl;
*/

}  // Namespace hip_impl.
