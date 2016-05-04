#ifndef LINETEMPLATE_FRAG_H_
#define LINETEMPLATE_FRAG_H_

#include <string>

namespace QueryRenderer {
struct LineTemplate_Frag {
  static const std::string source;
};

const std::string LineTemplate_Frag::source =
    "#version 450 core\n"
    "\n"
    "layout(location = 0) out vec4 color;\n"
    "layout(location = 1) out uint id;\n"
    "\n"
    "flat in uint fPrimitiveId;\n"
    "\n"
    "#define usePerVertColor <usePerVertColor>\n"
    "#if usePerVertColor == 1\n"
    "in vec4 fColor;\n"
    "#else\n"
    "flat in vec4 fColor;\n"
    "#endif\n"
    "\n"
    "void main() {\n"
    "    // TODO(croot): do some kind of temporary anti-aliasing?\n"
    "    color = fColor;\n"
    "    id = fPrimitiveId;\n"
    "}\n";

}  // namespace QueryRenderer

#endif  // LINETEMPLATE_FRAG_H_
