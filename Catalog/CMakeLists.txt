set(catalog_source_files
    Catalog.cpp
    Catalog.h
    LdapServer.cpp
    LdapServer.h
)

get_target_property(StringDictionary_BINARY_DIR StringDictionary BINARY_DIR)
include_directories(${StringDictionary_BINARY_DIR})

add_library(Catalog ${catalog_source_files})

if(ENABLE_CALCITE)
  target_link_libraries(Catalog SqliteConnector StringDictionary Fragmenter ldap Calcite)
else()
  target_link_libraries(Catalog SqliteConnector StringDictionary Fragmenter ldap)
endif()
