---
permalink: /navtreedata.js
---
/*
@licstart  The following is the entire license notice for the
JavaScript code in this file.

Copyright (C) 1997-2019 by Dimitri van Heesch

This program is free software; you can redistribute it and/or modify
it under the terms of version 2 of the GNU General Public License as published by
the Free Software Foundation

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

@licend  The above is the entire license notice
for the JavaScript code in this file
*/
var NAVTREE =
[
    [ "Home", "index.html", {% include_relative index.js %} ],
    [ "Download", "page_download.html", {% include_relative page_download.js %} ],
    [ "Available Tools", "page_tool.html", {% include_relative page_tool.js %} ],
    [ "Build Your Own Tool", "page_user_docs.html", {% include_relative page_user_docs.js %} ],
    [ "Obtaining Help and Reporting Bugs", "page_help.html", {% include_relative page_help.js %} ],
    [ "Developer Documentation", "page_dev_docs.html", {% include_relative page_dev_docs.js %} ],
    [ "License", "page_license.html", [
      [ "Primary DynamoRIO License: BSD", "page_license.html#sec_bsd", null ],
      [ "libelftc License", "page_license.html#sec_libelftc", null ],
      [ "Certain Extensions: LGPL 2.1", "page_license.html#sec_lgpl_licenses", null ],
      [ "Code Coverage genhtml: GPL 2", "page_license.html#sec_gpl_licenses", null ]
    ] ],
[ "API Reference", "files.html", [
    [ "Deprecated List", "deprecated.html", null ],
    [ "DynamoRIO Extension Details", "modules.html", {% include_relative modules.js %} ],
    [ "Data Structures", "annotated.html", [
      [ "Data Structures", "annotated.html", {% include_relative annotated_dup.js %} ],
      [ "Class Hierarchy", "hierarchy.html", {% include_relative hierarchy.js %} ],
      [ "Data Fields", "functions.html", [
        [ "All", "functions.html", {% include_relative functions_dup.js %} ],
        [ "Functions", "functions_func.html", null ],
        [ "Variables", "functions_vars.html", {% include_relative functions_vars.js %} ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", {% include_relative files_dup.js %} ],
      [ "Globals", "globals.html", [
        [ "All", "globals.html", {% include_relative globals_dup.js %} ],
        [ "Functions", "globals_func.html", {% include_relative globals_func.js %} ],
        [ "Variables", "globals_vars.html", null ],
        [ "Typedefs", "globals_type.html", null ],
        [ "Enumerations", "globals_enum.html", null ],
        [ "Enumerator", "globals_eval.html", {% include_relative globals_eval.js %} ],
        [ "Macros", "globals_defs.html", {% include_relative globals_defs.js %} ]
      ] ]
    ] ],  ] ]
];



var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';