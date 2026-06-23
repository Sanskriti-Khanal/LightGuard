"""Plain-English translations for EMBER2024 PE feature indices.

Provides:
  FEATURE_NAMES  — list of 2568 string names, one per feature index
  translate()    — converts a list of explain_prediction() dicts to readable sentences

Feature groups and their index ranges (from PEFeatureExtractor in thrember):
  general        [   0:    7]   7  scalar file properties
  histogram      [   7:  263] 256  byte-value frequency histogram
  byteentropy    [ 263:  519] 256  windowed byte-entropy joint histogram
  strings        [ 519:  696] 177  extracted printable-string statistics
  header         [ 696:  770]  74  PE COFF / Optional / DOS header fields
  section        [ 770:  994] 224  section table statistics (hash-bucketed)
  imports        [ 994: 2276] 1282  import table (hash-bucketed DLL + function names)
  exports        [2276: 2405] 129  export table (hash-bucketed function names)
  datadirectories[2405: 2439]  34  PE data directory virtual addresses and sizes
  richheader     [2439: 2472]  33  Rich header toolchain fingerprint (hash-bucketed)
  authenticode   [2472: 2480]   8  Authenticode digital signature attributes
  pefilewarnings [2480: 2568]  88  pefile parser warning flags
"""

from __future__ import annotations

# ── String regex feature keys (sorted, 77 total) ──────────────────────────────

_STRING_REGEX_KEYS: list[str] = [
    ".click(",
    "/EmbeddedFile",
    "/FlateDecode",
    "/URI",
    "/bin/",
    "/dev/",
    "/proc/",
    "/tmp/",
    "/usr/",
    "<script",
    "Invoke-Command",
    "Invoke-Expression",
    "Start-process",
    "base64",
    "base64string",
    "btc_wallet",
    "cache",
    "certificate",
    "clipboard",
    "command",
    "connect",
    "cookie",
    "create",
    "crypt",
    "debug",
    "decode",
    "delete",
    "desktop",
    "directory",
    "disk",
    "dos_msg",
    "download",
    "email_addr",
    "encode",
    "enum",
    "environment",
    "exit",
    "file",
    "file_path",
    "ftp",
    "get",
    "hidden",
    "hostname",
    "html",
    "http",
    "http://",
    "https://",
    "install",
    "internet",
    "ipv4_addr",
    "ipv6_addr",
    "javascript",
    "keyboard",
    "mac_addr",
    "memory",
    "module",
    "mutex",
    "onlick",
    "password",
    "post",
    "powershell",
    "privilege",
    "process",
    "registry_key",
    "remote",
    "resource",
    "security",
    "service",
    "shell",
    "snapshot",
    "system",
    "thread",
    "token",
    "url",
    "useragent",
    "wallet",
    "window",
]

# ── Header sub-feature names (in order from process_raw_features) ─────────────

_HEADER_IMAGE_CHARACTERISTICS = [
    "RELOCS_STRIPPED", "EXECUTABLE_IMAGE", "LINE_NUMS_STRIPPED",
    "LOCAL_SYMS_STRIPPED", "AGGRESIVE_WS_TRIM", "LARGE_ADDRESS_AWARE",
    "16BIT_MACHINE", "BYTES_REVERSED_LO", "32BIT_MACHINE", "DEBUG_STRIPPED",
    "REMOVABLE_RUN_FROM_SWAP", "NET_RUN_FROM_SWAP", "SYSTEM", "DLL",
    "UP_SYSTEM_ONLY", "BYTES_REVERSED_HI",
]
_HEADER_DLL_CHARACTERISTICS = [
    "HIGH_ENTROPY_VA", "DYNAMIC_BASE", "FORCE_INTEGRITY", "NX_COMPAT",
    "NO_ISOLATION", "NO_SEH", "NO_BIND", "APPCONTAINER",
    "WDM_DRIVER", "GUARD_CF", "TERMINAL_SERVER_AWARE",
]
_HEADER_DOS_MEMBERS = [
    "e_magic", "e_cblp", "e_cp", "e_crlc", "e_cparhdr",
    "e_minalloc", "e_maxalloc", "e_ss", "e_sp", "e_csum",
    "e_ip", "e_cs", "e_lfarlc", "e_ovno", "e_oemid",
    "e_oeminfo", "e_lfanew",
]

_DATA_DIR_NAMES = [
    "EXPORT", "IMPORT", "RESOURCE", "EXCEPTION", "SECURITY",
    "BASERELOC", "DEBUG", "COPYRIGHT", "GLOBALPTR", "TLS",
    "LOAD_CONFIG", "BOUND_IMPORT", "IAT", "DELAY_IMPORT",
    "COM_DESCRIPTOR", "RESERVED",
]

# ── Build the master feature name list ───────────────────────────────────────

def _build_feature_names() -> list[str]:
    names: list[str] = []

    # general [0:7]
    names += [
        "general_file_size",
        "general_file_entropy",
        "general_is_pe",
        "general_start_byte_0",
        "general_start_byte_1",
        "general_start_byte_2",
        "general_start_byte_3",
    ]

    # histogram [7:263]
    names += [f"histogram_byte_{i}" for i in range(256)]

    # byteentropy [263:519]
    names += [f"byteentropy_bin_{i}" for i in range(256)]

    # strings [519:696]
    names += [
        "strings_count",
        "strings_avg_length",
        "strings_printables_count",
    ]
    names += [f"strings_char_dist_{i}" for i in range(96)]
    names += ["strings_entropy"]
    names += [f"strings_regex_{k.replace('/', '_slash_').replace('.', '_dot_').replace('<', '_lt_').replace('(', '_lp_').replace('-', '_')}" for k in _STRING_REGEX_KEYS]

    # header [696:770]
    names += [
        "header_coff_timestamp",
        "header_coff_num_sections",
        "header_coff_num_symbols",
        "header_coff_sizeof_optional_header",
        "header_coff_pointer_to_symbol_table",
        "header_coff_machine_type",
        "header_optional_subsystem",
        "header_optional_major_image_version",
        "header_optional_minor_image_version",
        "header_optional_major_linker_version",
        "header_optional_minor_linker_version",
        "header_optional_major_os_version",
        "header_optional_minor_os_version",
        "header_optional_major_subsystem_version",
        "header_optional_minor_subsystem_version",
        "header_optional_sizeof_code",
        "header_optional_sizeof_headers",
        "header_optional_sizeof_image",
        "header_optional_sizeof_initialized_data",
        "header_optional_sizeof_uninitialized_data",
        "header_optional_sizeof_stack_reserve",
        "header_optional_sizeof_stack_commit",
        "header_optional_sizeof_heap_reserve",
        "header_optional_sizeof_heap_commit",
        "header_optional_address_of_entrypoint",
        "header_optional_base_of_code",
        "header_optional_image_base",
        "header_optional_section_alignment",
        "header_optional_checksum",
        "header_optional_num_rvas_and_sizes",
    ]
    names += [f"header_image_char_{c}" for c in _HEADER_IMAGE_CHARACTERISTICS]
    names += [f"header_dll_char_{c}" for c in _HEADER_DLL_CHARACTERISTICS]
    names += [f"header_dos_{m}" for m in _HEADER_DOS_MEMBERS]

    # section [770:994]
    names += [
        "section_num_sections",
        "section_num_zero_size",
        "section_num_empty_name",
        "section_num_rx",
        "section_num_writable",
        "section_max_entropy",
        "section_min_entropy",
        "section_max_size_ratio",
        "section_min_size_ratio",
        "section_max_vsize_ratio",
        "section_min_vsize_ratio",
    ]
    names += [f"section_size_hash_{i}" for i in range(50)]
    names += [f"section_vsize_hash_{i}" for i in range(50)]
    names += [f"section_entropy_hash_{i}" for i in range(50)]
    names += [f"section_char_hash_{i}" for i in range(50)]
    names += [f"section_entry_hash_{i}" for i in range(10)]
    names += [
        "section_overlay_size",
        "section_overlay_size_ratio",
        "section_overlay_entropy",
    ]

    # imports [994:2276]
    names += [
        "imports_num_dlls",
        "imports_num_functions",
    ]
    names += [f"imports_dll_hash_{i}" for i in range(256)]
    names += [f"imports_func_hash_{i}" for i in range(1024)]

    # exports [2276:2405]
    names += ["exports_num_functions"]
    names += [f"exports_func_hash_{i}" for i in range(128)]

    # datadirectories [2405:2439]
    for dd_name in _DATA_DIR_NAMES:
        names += [f"datadir_{dd_name}_size", f"datadir_{dd_name}_vaddr"]
    names += ["datadir_has_relocs", "datadir_has_dynamic_relocs"]

    # richheader [2439:2472]
    names += ["richheader_num_pairs"]
    names += [f"richheader_hash_{i}" for i in range(32)]

    # authenticode [2472:2480]
    names += [
        "authenticode_num_certs",
        "authenticode_self_signed",
        "authenticode_empty_program_name",
        "authenticode_no_countersigner",
        "authenticode_parse_error",
        "authenticode_chain_max_depth",
        "authenticode_latest_signing_time",
        "authenticode_signing_time_diff",
    ]

    # pefilewarnings [2480:2568]
    names += [f"pefilewarning_{i}" for i in range(87)]
    names += ["pefilewarnings_num_warnings"]

    assert len(names) == 2568, f"Expected 2568 feature names, got {len(names)}"
    return names


FEATURE_NAMES: list[str] = _build_feature_names()


# ── Human-readable explanations keyed by feature name (or prefix) ─────────────
#
# For scalar/named features the key is the exact feature name.
# For hash-bucketed groups the key is the group prefix (matched by startswith).
# Values are sentence templates; {value:.3g} and {shap:+.3f} are available.

_EXACT: dict[str, str] = {
    # general
    "general_file_size":        "File size is {value:.0f} bytes",
    "general_file_entropy":     "Overall file entropy is {value:.3f} (higher = more encrypted/packed)",
    "general_is_pe":            "File {'is' if value else 'is not'} a valid PE executable",
    "general_start_byte_0":     "First byte of file is 0x{value:02X}",
    "general_start_byte_1":     "Second byte is 0x{value:02X}",
    "general_start_byte_2":     "Third byte is 0x{value:02X}",
    "general_start_byte_3":     "Fourth byte is 0x{value:02X}",
    # strings
    "strings_count":            "Contains {value:.0f} printable strings",
    "strings_avg_length":       "Average string length is {value:.1f} characters",
    "strings_printables_count": "Total printable character count is {value:.0f}",
    "strings_entropy":          "String section entropy is {value:.3f}",
    # header scalar
    "header_coff_timestamp":    "PE compile timestamp is {value:.0f} (Unix epoch)",
    "header_coff_num_sections": "PE file has {value:.0f} sections",
    "header_coff_num_symbols":  "Symbol table has {value:.0f} symbols",
    "header_coff_machine_type": "Target machine type code is {value:.0f}",
    "header_optional_subsystem": "Windows subsystem code is {value:.0f}",
    "header_optional_sizeof_code": "Size of .text (code) section is {value:.0f} bytes",
    "header_optional_sizeof_image": "In-memory image size is {value:.0f} bytes",
    "header_optional_sizeof_initialized_data": "Initialized data size is {value:.0f} bytes",
    "header_optional_sizeof_uninitialized_data": "Uninitialized (BSS) data size is {value:.0f} bytes",
    "header_optional_sizeof_stack_reserve": "Stack reserve size is {value:.0f} bytes",
    "header_optional_address_of_entrypoint": "Entry point RVA is 0x{value:X}",
    "header_optional_image_base": "Preferred image base address is 0x{value:X}",
    "header_optional_checksum": "PE checksum value is {value:.0f}",
    # section stats
    "section_num_sections":     "File has {value:.0f} PE sections",
    "section_num_zero_size":    "{value:.0f} sections have zero raw size",
    "section_num_empty_name":   "{value:.0f} sections have empty names",
    "section_num_rx":           "{value:.0f} sections are readable+executable",
    "section_num_writable":     "{value:.0f} sections are writable",
    "section_max_entropy":      "Highest section entropy is {value:.3f}",
    "section_min_entropy":      "Lowest section entropy is {value:.3f}",
    "section_overlay_size":     "Overlay data (appended after PE) is {value:.0f} bytes",
    "section_overlay_entropy":  "Overlay entropy is {value:.3f}",
    # imports/exports counts
    "imports_num_dlls":         "Imports from {value:.0f} DLLs",
    "imports_num_functions":    "Imports {value:.0f} functions total",
    "exports_num_functions":    "Exports {value:.0f} functions",
    # datadirectories
    "datadir_has_relocs":       "File {'has' if value else 'lacks'} relocation table",
    "datadir_has_dynamic_relocs": "File {'has' if value else 'lacks'} dynamic relocation data",
    "datadir_SECURITY_size":    "Security directory (Authenticode) size is {value:.0f} bytes",
    "datadir_TLS_size":         "TLS directory size is {value:.0f} bytes (non-zero = uses TLS callbacks)",
    "datadir_IAT_size":         "Import Address Table size is {value:.0f} bytes",
    # richheader
    "richheader_num_pairs":     "Rich header has {value:.0f} toolchain version pairs",
    # authenticode
    "authenticode_num_certs":   "Authenticode certificate chain has {value:.0f} certificate(s)",
    "authenticode_self_signed": "Authenticode certificate {'is' if value else 'is not'} self-signed",
    "authenticode_empty_program_name": "Authenticode program name {'is empty' if value else 'is present'}",
    "authenticode_no_countersigner": "Authenticode {'lacks' if value else 'has'} a countersigner timestamp",
    "authenticode_parse_error": "Authenticode signature {'failed' if value else 'parsed'} to parse",
    "authenticode_chain_max_depth": "Certificate chain depth is {value:.0f}",
    "authenticode_signing_time_diff": "Gap between signing time and compile time is {value:.0f} seconds",
    # pefile warnings count
    "pefilewarnings_num_warnings": "{value:.0f} pefile parser warnings found",
}

# String regex features get individual explanations
_STRING_REGEX_HUMAN: dict[str, str] = {
    ".click(":          "contains click-event JavaScript injection strings",
    "/EmbeddedFile":    "contains embedded-file PDF object markers",  # TODO: verify PDF relevance
    "/FlateDecode":     "contains FlateDecode PDF compression markers",  # TODO: verify
    "/URI":             "contains PDF /URI action markers",  # TODO: verify
    "/bin/":            "references Unix /bin/ paths",
    "/dev/":            "references Unix /dev/ device paths",
    "/proc/":           "references Linux /proc/ filesystem paths",
    "/tmp/":            "references Unix /tmp/ directory",
    "/usr/":            "references Unix /usr/ directory",
    "<script":          "contains HTML <script> tags",
    "Invoke-Command":   "contains PowerShell Invoke-Command strings",
    "Invoke-Expression": "contains PowerShell Invoke-Expression (iex) strings",
    "Start-process":    "contains PowerShell Start-Process strings",
    "base64":           "contains base64 encoding references",
    "base64string":     "contains base64-encoded string patterns",
    "btc_wallet":       "contains Bitcoin wallet address patterns",
    "cache":            "references cache-related strings",
    "certificate":      "references certificate-related strings",
    "clipboard":        "references clipboard access strings",
    "command":          "references command execution strings",
    "connect":          "references network connect strings",
    "cookie":           "references HTTP cookie strings",
    "create":           "references object/process creation strings",
    "crypt":            "references cryptography-related strings",
    "debug":            "references debugger/debug strings",
    "decode":           "references data-decoding strings",
    "delete":           "references file/object deletion strings",
    "desktop":          "references desktop/user environment strings",
    "directory":        "references directory manipulation strings",
    "disk":             "references disk/storage strings",
    "dos_msg":          "contains MS-DOS stub error message",
    "download":         "references download/retrieval strings",
    "email_addr":       "contains email address patterns",
    "encode":           "references data-encoding strings",
    "enum":             "references system enumeration strings",
    "environment":      "references environment variable strings",
    "exit":             "references process exit strings",
    "file":             "references file operation strings",
    "file_path":        "contains Windows file path patterns",
    "ftp":              "references FTP protocol strings",
    "get":              "references HTTP GET or getter strings",
    "hidden":           "references hidden attribute or stealth strings",
    "hostname":         "references hostname lookup strings",
    "html":             "references HTML markup strings",
    "http":             "references HTTP protocol strings",
    "http://":          "contains plaintext HTTP URLs",
    "https://":         "contains HTTPS URLs",
    "install":          "references installation strings",
    "internet":         "references internet connectivity strings",
    "ipv4_addr":        "contains IPv4 address patterns",
    "ipv6_addr":        "contains IPv6 address patterns",
    "javascript":       "references JavaScript strings",
    "keyboard":         "references keyboard input strings (possible keylogger)",
    "mac_addr":         "contains MAC address patterns",
    "memory":           "references memory manipulation strings",
    "module":           "references module loading strings",
    "mutex":            "references mutex/lock object strings",
    "onlick":           "contains onclick event handler strings",  # note: typo in thrember regex key
    "password":         "references password-related strings",
    "post":             "references HTTP POST strings",
    "powershell":       "references PowerShell execution strings",
    "privilege":        "references privilege escalation strings",
    "process":          "references process manipulation strings",
    "registry_key":     "contains Windows registry key path patterns",
    "remote":           "references remote access strings",
    "resource":         "references resource loading strings",
    "security":         "references security bypass or audit strings",
    "service":          "references Windows service strings",
    "shell":            "references shell execution strings",
    "snapshot":         "references snapshot or screenshot strings",
    "system":           "references system-level operation strings",
    "thread":           "references thread manipulation strings",
    "token":            "references security token strings",
    "url":              "contains URL patterns",
    "useragent":        "contains HTTP User-Agent header strings",
    "wallet":           "references cryptocurrency wallet strings",
    "window":           "references window manipulation strings",
}

# Group-level fallback descriptions (matched by feature name prefix)
_GROUP_FALLBACK: dict[str, str] = {
    "histogram_byte_":     "byte frequency histogram bucket",
    "byteentropy_bin_":    "byte-entropy joint histogram bucket",
    "strings_char_dist_":  "printable character frequency bucket",
    "strings_regex_":      "string pattern match count",
    "header_image_char_":  "PE image characteristic flag",
    "header_dll_char_":    "PE DLL characteristic flag",
    "header_dos_":         "DOS header field value",
    "section_size_hash_":  "section name↔size hash bucket",
    "section_vsize_hash_": "section name↔virtual-size hash bucket",
    "section_entropy_hash_": "section name↔entropy hash bucket",
    "section_char_hash_":  "section characteristics hash bucket",
    "section_entry_hash_": "entry-point section name hash bucket",
    "imports_dll_hash_":   "imported DLL name hash bucket",
    "imports_func_hash_":  "imported function name hash bucket",
    "exports_func_hash_":  "exported function name hash bucket",
    "datadir_":            "PE data directory field",
    "richheader_hash_":    "Rich header toolchain hash bucket",
    "pefilewarning_":      "pefile parser warning flag",
}

# Sentence prefixes that indicate risk direction for malware classification
_RISK_HIGH = "High-risk indicator"
_RISK_LOW  = "Low-risk indicator"


def feature_description(name: str, raw_value: float) -> str:
    """Return a plain-English description for a single feature name + value."""
    # Exact match
    if name in _EXACT:
        tmpl = _EXACT[name]
        try:
            return eval(f'f"""{tmpl}"""', {"value": raw_value})  # noqa: S307 — controlled template
        except Exception:
            return f"{name} = {raw_value:.4g}"

    # String regex features: map back to the original key
    if name.startswith("strings_regex_"):
        # Find the original key by matching the sanitised suffix
        idx = None
        for i, key in enumerate(_STRING_REGEX_KEYS):
            sanitised = key.replace("/", "_slash_").replace(".", "_dot_").replace("<", "_lt_").replace("(", "_lp_").replace("-", "_")
            if name == f"strings_regex_{sanitised}":
                idx = i
                break
        if idx is not None:
            original_key = _STRING_REGEX_KEYS[idx]
            human = _STRING_REGEX_HUMAN.get(original_key, f"matches pattern '{original_key}'")
            return f"String matches ({raw_value:.0f} occurrences): {human}"

    # Group-level fallback
    for prefix, desc in _GROUP_FALLBACK.items():
        if name.startswith(prefix):
            return f"{desc} (value={raw_value:.4g})"

    return f"{name} = {raw_value:.4g}"


def translate(top_features: list[dict]) -> list[str]:
    """Convert explain_prediction() output into plain-English sentences.

    Args:
        top_features: list of dicts from explain_prediction(), each containing
            keys 'feature_idx', 'feature_name', 'shap_value', 'raw_value'.

    Returns:
        List of human-readable sentences, one per feature, ordered by |SHAP|.
    """
    sentences: list[str] = []
    for entry in top_features:
        name = entry["feature_name"]
        raw = entry["raw_value"]
        shap = entry["shap_value"]

        direction = _RISK_HIGH if shap > 0 else _RISK_LOW
        desc = feature_description(name, raw)
        sentences.append(f"{direction} (SHAP {shap:+.3f}): {desc}")

    return sentences
