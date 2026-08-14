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

import datetime

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


# ── Machine type and subsystem code maps ──────────────────────────────────────

_MACHINE_TYPE: dict[int, str] = {
    332:   "32-bit (x86) Windows executable",
    34404: "64-bit (x64) Windows executable",
    512:   "64-bit Itanium (IA-64) executable",
    43620: "64-bit ARM (ARM64) executable",
    452:   "32-bit ARM executable",
}

_SUBSYSTEM: dict[int, str] = {
    0:  "unknown subsystem",
    1:  "native Windows driver/service",
    2:  "GUI desktop application",
    3:  "console (command-line) application",
    5:  "OS/2 console application",
    7:  "POSIX console application",
    9:  "Windows CE GUI application",
    10: "EFI application",
    14: "EFI ROM image",
    16: "Xbox application",
    17: "Windows boot application",
}

# Human-readable labels for image characteristics flags
_IMAGE_CHAR_HUMAN: dict[str, str] = {
    "RELOCS_STRIPPED":        "Relocation info stripped (fixed load address)",
    "EXECUTABLE_IMAGE":       "Marked as executable",
    "LINE_NUMS_STRIPPED":     "Debug line numbers removed",
    "LOCAL_SYMS_STRIPPED":    "Local debug symbols removed",
    "AGGRESIVE_WS_TRIM":      "Aggressive working-set trimming flag set",
    "LARGE_ADDRESS_AWARE":    "Can use more than 2 GB of memory",
    "16BIT_MACHINE":          "Flagged as 16-bit machine (very unusual)",
    "32BIT_MACHINE":          "Flagged as 32-bit machine",
    "DEBUG_STRIPPED":         "Debug info stored separately",
    "REMOVABLE_RUN_FROM_SWAP": "Runs from swap on removable media",
    "NET_RUN_FROM_SWAP":      "Runs from swap over network",
    "SYSTEM":                 "Marked as a system file",
    "DLL":                    "Marked as a DLL (not a standalone program)",
    "UP_SYSTEM_ONLY":         "Only runs on single-processor systems",
}

# Human-readable labels for DLL characteristics flags
_DLL_CHAR_HUMAN: dict[str, str] = {
    "HIGH_ENTROPY_VA":        "High-entropy address space layout (extra ASLR)",
    "DYNAMIC_BASE":           "Address space layout randomisation (ASLR) enabled",
    "FORCE_INTEGRITY":        "Code-integrity checks enforced",
    "NX_COMPAT":              "Data Execution Prevention (DEP/NX) compatible",
    "NO_ISOLATION":           "Manifest isolation disabled",
    "NO_SEH":                 "Structured exception handling disabled",
    "NO_BIND":                "Binding to imported addresses disabled",
    "APPCONTAINER":           "Runs in an AppContainer sandbox",
    "WDM_DRIVER":             "WDM (kernel) driver",
    "GUARD_CF":               "Control Flow Guard (CFG) enabled",
    "TERMINAL_SERVER_AWARE":  "Designed for multi-user (Terminal Services) use",
}

# ── String regex human descriptions ──────────────────────────────────────────

_STRING_REGEX_HUMAN: dict[str, str] = {
    ".click(":          "click-event JavaScript injection strings",
    "/EmbeddedFile":    "embedded-file markers (PDF-like content)",
    "/FlateDecode":     "FlateDecode compression markers",
    "/URI":             "URI action markers",
    "/bin/":            "Unix /bin/ path references",
    "/dev/":            "Unix /dev/ device path references",
    "/proc/":           "Linux /proc/ filesystem references",
    "/tmp/":            "Unix /tmp/ directory references",
    "/usr/":            "Unix /usr/ directory references",
    "<script":          "HTML script tags",
    "Invoke-Command":   "PowerShell Invoke-Command strings",
    "Invoke-Expression": "PowerShell Invoke-Expression (iex) strings",
    "Start-process":    "PowerShell Start-Process strings",
    "base64":           "base64-encoded data",
    "base64string":     "base64 string patterns",
    "btc_wallet":       "Bitcoin wallet address patterns",
    "cache":            "cache-related strings",
    "certificate":      "certificate-related strings",
    "clipboard":        "clipboard access strings",
    "command":          "command execution strings",
    "connect":          "network connection strings",
    "cookie":           "HTTP cookie strings",
    "create":           "object or process creation strings",
    "crypt":            "cryptographic operation strings",
    "debug":            "debugger or debug strings",
    "decode":           "data-decoding strings",
    "delete":           "file or object deletion strings",
    "desktop":          "desktop or user environment strings",
    "directory":        "directory manipulation strings",
    "disk":             "disk or storage strings",
    "dos_msg":          "MS-DOS stub message",
    "download":         "file download strings",
    "email_addr":       "email address patterns",
    "encode":           "data-encoding strings",
    "enum":             "system enumeration strings",
    "environment":      "environment variable strings",
    "exit":             "process exit strings",
    "file":             "file operation strings",
    "file_path":        "Windows file path patterns",
    "ftp":              "FTP protocol strings",
    "get":              "HTTP GET strings",
    "hidden":           "hidden attribute or stealth strings",
    "hostname":         "hostname lookup strings",
    "html":             "HTML markup strings",
    "http":             "HTTP protocol strings",
    "http://":          "plain (unencrypted) HTTP web addresses",
    "https://":         "HTTPS web addresses",
    "install":          "installation strings",
    "internet":         "internet connectivity strings",
    "ipv4_addr":        "IPv4 address patterns",
    "ipv6_addr":        "IPv6 address patterns",
    "javascript":       "JavaScript strings",
    "keyboard":         "keyboard input strings (possible keylogger trait)",
    "mac_addr":         "MAC address patterns",
    "memory":           "memory manipulation strings",
    "module":           "module loading strings",
    "mutex":            "mutex or lock strings",
    "onlick":           "click event handler strings",
    "password":         "password-related strings",
    "post":             "HTTP POST strings",
    "powershell":       "PowerShell execution strings",
    "privilege":        "privilege escalation strings",
    "process":          "process manipulation strings",
    "registry_key":     "Windows registry path patterns",
    "remote":           "remote access strings",
    "resource":         "resource loading strings",
    "security":         "security bypass or audit strings",
    "service":          "Windows service strings",
    "shell":            "shell execution strings",
    "snapshot":         "snapshot or screenshot strings",
    "system":           "system-level operation strings",
    "thread":           "thread manipulation strings",
    "token":            "security token strings",
    "url":              "URL patterns",
    "useragent":        "HTTP User-Agent header strings",
    "wallet":           "cryptocurrency wallet strings",
    "window":           "window manipulation strings",
}


# ── Value-aware plain-English description ────────────────────────────────────

def feature_description(name: str, raw_value: float) -> str:
    """Return a plain-English description for a single feature name + value."""
    v = raw_value

    # ── general ──────────────────────────────────────────────────────────────
    if name == "general_file_size":
        kb = v / 1024
        if kb < 5:
            size_note = f"{v:.0f} bytes (unusually small)"
        elif kb < 1024:
            size_note = f"{kb:.0f} KB"
        else:
            size_note = f"{kb/1024:.1f} MB"
        return f"File size is {size_note}"

    if name == "general_file_entropy":
        if v > 7.2:
            return f"File entropy is very high ({v:.2f}/8) — likely packed or encrypted"
        if v > 6.0:
            return f"File entropy is elevated ({v:.2f}/8) — may contain compressed data"
        if v < 2.0:
            return f"File entropy is very low ({v:.2f}/8) — mostly zeroed or sparse"
        return f"File entropy is normal ({v:.2f}/8)"

    if name == "general_is_pe":
        return "Valid PE executable format" if v else "Not a valid PE executable"

    if name.startswith("general_start_byte_"):
        pos = name.split("_")[-1]
        return f"File starts with byte 0x{int(v):02X} at position {pos}"

    # ── strings ───────────────────────────────────────────────────────────────
    if name == "strings_count":
        n = int(v)
        if n == 0:
            return "No readable text strings embedded (unusual)"
        if n < 10:
            return f"Very few readable text strings ({n})"
        return f"Contains {n:,} readable text strings"

    if name == "strings_avg_length":
        return f"Average embedded string length is {v:.0f} characters"

    if name == "strings_printables_count":
        return f"Total printable character count is {v:,.0f}"

    if name == "strings_entropy":
        if v > 5.5:
            return f"Embedded strings are highly varied or obfuscated (entropy {v:.2f})"
        return f"String section entropy is {v:.2f}"

    # ── header scalars ────────────────────────────────────────────────────────
    if name == "header_coff_machine_type":
        label = _MACHINE_TYPE.get(int(v))
        if label:
            return f"Built as a {label}"
        return "Unusual or uncommon target architecture"

    if name == "header_optional_subsystem":
        label = _SUBSYSTEM.get(int(v))
        if label:
            return f"Windows subsystem: {label}"
        return f"Unusual Windows subsystem code ({int(v)})"

    if name == "header_coff_timestamp":
        if v <= 0:
            return "Compile timestamp is missing or zeroed"
        try:
            year = datetime.datetime.fromtimestamp(v, tz=datetime.timezone.utc).year
            if year < 2000:
                return f"Compile timestamp is very old ({year}) — may be forged"
            if year > datetime.datetime.now(datetime.timezone.utc).year + 1:
                return f"Compile timestamp is in the future ({year}) — likely forged"
            return f"Compiled around {year}"
        except (OSError, OverflowError):
            return "Compile timestamp is out of normal range"

    if name == "header_coff_num_sections":
        n = int(v)
        if n == 0:
            return "No PE sections (abnormal)"
        if n > 15:
            return f"Unusually many PE sections ({n})"
        return f"File has {n} PE sections"

    if name == "header_coff_num_symbols":
        return ("No debug symbol table" if v == 0
                else f"Debug symbol table has {v:.0f} entries (unusual in release builds)")

    if name == "header_optional_sizeof_code":
        return f"Code section is {v/1024:.0f} KB"

    if name == "header_optional_sizeof_image":
        return f"Loaded image size is {v/1024:.0f} KB"

    if name == "header_optional_sizeof_initialized_data":
        return f"Initialised data is {v/1024:.0f} KB"

    if name == "header_optional_sizeof_uninitialized_data":
        return (f"Uninitialised data (BSS) is {v/1024:.0f} KB"
                if v > 0 else "No uninitialised data (BSS)")

    if name == "header_optional_sizeof_stack_reserve":
        mb = v / (1024 * 1024)
        if mb > 64:
            return f"Stack reserve is very large ({mb:.0f} MB) — unusual"
        return f"Stack reserve is {v/1024:.0f} KB"

    if name == "header_optional_address_of_entrypoint":
        return f"Entry point address is 0x{int(v):X}"

    if name == "header_optional_image_base":
        std = {0x400000, 0x1000000, 0x10000000, 0x180000000}
        base = int(v)
        if base in std:
            return f"Standard image base address (0x{base:X})"
        return f"Non-standard image base address (0x{base:X})"

    if name == "header_optional_checksum":
        return ("No PE checksum (common in non-driver files)"
                if v == 0 else f"PE checksum is present (0x{int(v):X})")

    # ── header flags ──────────────────────────────────────────────────────────
    if name.startswith("header_image_char_"):
        flag = name[len("header_image_char_"):]
        human = _IMAGE_CHAR_HUMAN.get(flag)
        if human:
            return human if v else f"Not flagged: {human.lower()}"
        return f"PE image flag {flag} is {'set' if v else 'not set'}"

    if name.startswith("header_dll_char_"):
        flag = name[len("header_dll_char_"):]
        human = _DLL_CHAR_HUMAN.get(flag)
        if human:
            return human if v else f"Missing security feature: {human.lower()}"
        return f"PE DLL characteristic {flag} is {'enabled' if v else 'disabled'}"

    if name.startswith("header_dos_"):
        field = name[len("header_dos_"):]
        if field == "e_magic":
            return ("Valid MZ magic number (normal PE header)"
                    if int(v) == 0x5A4D else f"Unusual MZ magic value (0x{int(v):X})")
        return f"DOS header field {field} = {v:.4g}"

    # ── section stats ─────────────────────────────────────────────────────────
    if name == "section_num_sections":
        n = int(v)
        return (f"File has {n} PE sections"
                if 1 <= n <= 10 else f"Unusual section count ({n})")

    if name == "section_num_zero_size":
        return (f"{int(v)} section(s) have zero size (unusual)"
                if v > 0 else "All sections have non-zero size")

    if name == "section_num_empty_name":
        return (f"{int(v)} section(s) have empty names"
                if v > 0 else "All sections have names")

    if name == "section_num_rx":
        n = int(v)
        return (f"{n} readable+executable section(s)"
                if n <= 2 else f"{n} readable+executable sections (unusually many)")

    if name == "section_num_writable":
        n = int(v)
        return ("No writable sections (standard)"
                if n == 0 else f"{n} writable section(s) — unusual in most programs")

    if name == "section_max_entropy":
        if v > 7.5:
            return f"A section has very high entropy ({v:.2f}/8) — strongly suggests packing or encryption"
        if v > 6.5:
            return f"A section has elevated entropy ({v:.2f}/8) — may contain compressed data"
        return f"Highest section entropy is {v:.2f}/8 (normal range)"

    if name == "section_min_entropy":
        return f"Lowest section entropy is {v:.2f}/8"

    if name == "section_overlay_size":
        return (f"Has {v/1024:.0f} KB of data appended after the PE structure (overlay)"
                if v > 0 else "No data appended after PE structure")

    if name == "section_overlay_entropy":
        if v > 7.0:
            return f"Appended overlay data has very high entropy ({v:.2f}/8) — likely encrypted"
        return f"Overlay data entropy is {v:.2f}/8"

    # ── imports / exports ──────────────────────────────────────────────────────
    if name == "imports_num_dlls":
        n = int(v)
        if n == 0:
            return "Imports no libraries (unusual — may unpack dynamically)"
        if n > 50:
            return f"Imports from {n} libraries (unusually many)"
        return f"Imports from {n} librar{'y' if n == 1 else 'ies'}"

    if name == "imports_num_functions":
        n = int(v)
        if n == 0:
            return "Imports no functions (unusual)"
        return f"Imports {n:,} API function{'s' if n != 1 else ''}"

    if name == "exports_num_functions":
        n = int(v)
        return ("Exports no functions (normal for EXE files)"
                if n == 0 else f"Exports {n} function{'s' if n != 1 else ''}")

    # ── authenticode ──────────────────────────────────────────────────────────
    if name == "authenticode_latest_signing_time":
        if v <= 0:
            return "No digital signature timestamp"
        try:
            year = datetime.datetime.fromtimestamp(v, tz=datetime.timezone.utc).year
            now_year = datetime.datetime.now(datetime.timezone.utc).year
            if year >= now_year - 2:
                return f"Has a recent digital signature (signed {year})"
            if year >= 2010:
                return f"Has a digital signature from {year}"
            return f"Digital signature timestamp is old ({year})"
        except (OSError, OverflowError):
            return "Digital signature timestamp is out of normal range"

    if name == "authenticode_num_certs":
        n = int(v)
        if n == 0:
            return "No digital certificate chain (unsigned)"
        return f"Digital certificate chain has {n} certificate{'s' if n != 1 else ''}"

    if name == "authenticode_self_signed":
        return ("Has a self-signed certificate (not from a trusted publisher)"
                if v else "Certificate is from a trusted signing authority")

    if name == "authenticode_empty_program_name":
        return ("Publisher name is missing from the digital signature"
                if v else "Publisher name is present in the digital signature")

    if name == "authenticode_no_countersigner":
        return ("No trusted timestamp from a countersigner"
                if v else "Has a countersigned timestamp (signature is time-stamped)")

    if name == "authenticode_parse_error":
        return ("Digital signature is missing or could not be parsed"
                if v else "Digital signature parsed successfully")

    if name == "authenticode_chain_max_depth":
        return f"Certificate chain depth is {int(v)}"

    if name == "authenticode_signing_time_diff":
        days = abs(v) / 86400
        if days > 365:
            return f"Signing time differs from compile time by {days/365:.1f} years (unusual)"
        if days > 30:
            return f"Signing time differs from compile time by {days:.0f} days"
        return f"Signing and compile times are close ({days:.0f} days apart)"

    # ── data directories ──────────────────────────────────────────────────────
    if name == "datadir_has_relocs":
        return ("Has relocation table (can be loaded at any address)"
                if v else "No relocation table (must load at fixed address)")

    if name == "datadir_has_dynamic_relocs":
        return ("Has dynamic relocation data (ASLR-compatible)"
                if v else "No dynamic relocation data")

    if name == "datadir_SECURITY_size":
        return (f"Authenticode security directory is {v:.0f} bytes"
                if v > 0 else "No Authenticode security directory (unsigned)")

    if name == "datadir_TLS_size":
        return ("Has TLS callbacks (code runs before main entry point)"
                if v > 0 else "No TLS callbacks")

    if name == "datadir_IAT_size":
        return f"Import Address Table is {v:.0f} bytes"

    if name.startswith("datadir_") and name.endswith("_size"):
        dd = name[len("datadir_"):-len("_size")]
        return (f"{dd} directory is present ({v:.0f} bytes)"
                if v > 0 else f"No {dd} directory")

    # ── rich header ───────────────────────────────────────────────────────────
    if name == "richheader_num_pairs":
        n = int(v)
        return (f"Rich header has {n} compiler toolchain entr{'y' if n == 1 else 'ies'}"
                if n > 0 else "No Rich header (compiler fingerprint absent or stripped)")

    # ── pefile warnings ───────────────────────────────────────────────────────
    if name == "pefilewarnings_num_warnings":
        n = int(v)
        return ("File structure is clean and well-formed — no parser warnings"
                if n == 0 else f"{n} file structure warning{'s' if n != 1 else ''} detected")

    if name.startswith("pefilewarning_"):
        return ("File structure check passed"
                if v == 0 else "File structure check flagged an issue")

    # ── string regex features ─────────────────────────────────────────────────
    if name.startswith("strings_regex_"):
        for i, key in enumerate(_STRING_REGEX_KEYS):
            sanitised = (key.replace("/", "_slash_").replace(".", "_dot_")
                           .replace("<", "_lt_").replace("(", "_lp_").replace("-", "_"))
            if name == f"strings_regex_{sanitised}":
                human = _STRING_REGEX_HUMAN.get(key, f"pattern '{key}'")
                n = int(v)
                if n == 0:
                    return f"No {human} found"
                return f"Contains {human} ({n} occurrence{'s' if n != 1 else ''})"

    # ── hash-bucketed groups — best-effort ────────────────────────────────────
    if name.startswith("histogram_byte_"):
        return "Unusual byte frequency distribution"

    if name.startswith("byteentropy_bin_"):
        return "Unusual byte-entropy pattern"

    if name.startswith("strings_char_dist_"):
        return "Unusual character distribution in strings"

    if name.startswith("imports_dll_hash_"):
        return "Unusual imported library pattern"

    if name.startswith("imports_func_hash_"):
        return "Unusual imported function pattern"

    if name.startswith("exports_func_hash_"):
        return "Unusual exported function pattern"

    if name.startswith("section_size_hash_"):
        return "Unusual section size pattern"

    if name.startswith("section_vsize_hash_"):
        return "Unusual section virtual-size pattern"

    if name.startswith("section_entropy_hash_"):
        return "Unusual section entropy pattern"

    if name.startswith("section_char_hash_"):
        return "Unusual section characteristics pattern"

    if name.startswith("section_entry_hash_"):
        return "Unusual entry-point section name"

    if name.startswith("richheader_hash_"):
        return "Unusual compiler or toolchain fingerprint"

    # final fallback
    return "Unusual structural trait"


def translate(top_features: list[dict], verbose: bool = False) -> list[str]:
    """Convert explain_prediction() output into plain-English sentences.

    Args:
        top_features: list of dicts from explain_prediction(), each containing
            keys 'feature_idx', 'feature_name', 'shap_value', 'raw_value'.
        verbose: if True, include the SHAP value in the sentence so callers
            (e.g. _parse_reasons in the Flask app) can extract it for bar widths.
            If False (default), produce clean user-facing sentences with no numbers.

    Returns:
        List of human-readable sentences, one per feature, ordered by |SHAP|.
    """
    _RISK_HIGH = "High-risk indicator" if verbose else "High-risk"
    _RISK_LOW  = "Low-risk indicator"  if verbose else "Low-risk"

    sentences: list[str] = []
    for entry in top_features:
        name  = entry["feature_name"]
        raw   = entry["raw_value"]
        shap  = entry["shap_value"]

        direction = _RISK_HIGH if shap > 0 else _RISK_LOW
        desc      = feature_description(name, raw)

        if verbose:
            sentences.append(f"{direction} (SHAP {shap:+.3f}): {desc}")
        else:
            sentences.append(f"{direction}: {desc}")

    return sentences
