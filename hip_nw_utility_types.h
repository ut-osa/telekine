#ifndef __HIP_NW_UTILITY_TYPES_H__
#define __HIP_NW_UTILITY_TYPES_H__

struct hipFuncAttributes;
typedef struct hipFuncAttributes hipFuncAttributes;
typedef uint16_t Elf_Half;
typedef uint32_t Elf_Word;
typedef int32_t Elf_Sword;
typedef uint64_t Elf_Xword;
typedef int64_t Elf_Sxword;
typedef uint32_t Elf32_Addr;
typedef uint32_t Elf32_Off;
typedef uint64_t Elf64_Addr;
typedef uint64_t Elf64_Off;
#define Elf32_Half Elf_Half
#define Elf64_Half Elf_Half
#define Elf32_Word Elf_Word
#define Elf64_Word Elf_Word
#define Elf32_Sword Elf_Sword
#define Elf64_Sword Elf_Sword
#define EI_NIDENT 16
typedef struct Elf64_Ehdr {
    unsigned char e_ident[EI_NIDENT];
    Elf_Half e_type;
    Elf_Half e_machine;
    Elf_Word e_version;
    Elf64_Addr e_entry;
    Elf64_Off e_phoff;
    Elf64_Off e_shoff;
    Elf_Word e_flags;
    Elf_Half e_ehsize;
    Elf_Half e_phentsize;
    Elf_Half e_phnum;
    Elf_Half e_shentsize;
    Elf_Half e_shnum;
    Elf_Half e_shstrndx;
} Elf64_Ehdr;

#endif                                           // ndef __HIP_NW_UTILITY_TYPES_H__
