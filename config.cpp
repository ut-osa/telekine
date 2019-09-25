#include "config.h"

namespace tlkine {

global_config &__config()
{
   static global_config _config;
   return _config;
}

global_config &config = __config();

}
