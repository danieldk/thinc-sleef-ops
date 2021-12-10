from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr

cdef extern from "libcpuid.h":
    const int CPU_FLAGS_MAX = 128

    struct cpu_id_t:
        uint8_t flags[CPU_FLAGS_MAX]

    struct cpu_raw_data_t:
        pass

    int cpuid_get_raw_data(cpu_raw_data_t *)
    const char *cpu_feature_str(cpu_feature_t feature)
    int cpu_identify(cpu_raw_data_t *, cpu_id_t *)
    const char *cpuid_error();
    int cpuid_present()

    ctypedef enum cpu_feature_t:
        CPU_FEATURE_FPU = 0
        CPU_FEATURE_VME
        CPU_FEATURE_DE
        CPU_FEATURE_PSE
        CPU_FEATURE_TSC
        CPU_FEATURE_MSR
        CPU_FEATURE_PAE
        CPU_FEATURE_MCE
        CPU_FEATURE_CX8
        CPU_FEATURE_APIC
        CPU_FEATURE_MTRR
        CPU_FEATURE_SEP
        CPU_FEATURE_PGE
        CPU_FEATURE_MCA
        CPU_FEATURE_CMOV
        CPU_FEATURE_PAT
        CPU_FEATURE_PSE36
        CPU_FEATURE_PN
        CPU_FEATURE_CLFLUSH
        CPU_FEATURE_DTS
        CPU_FEATURE_ACPI
        CPU_FEATURE_MMX
        CPU_FEATURE_FXSR
        CPU_FEATURE_SSE
        CPU_FEATURE_SSE2
        CPU_FEATURE_SS
        CPU_FEATURE_HT
        CPU_FEATURE_TM
        CPU_FEATURE_IA64
        CPU_FEATURE_PBE
        CPU_FEATURE_PNI
        CPU_FEATURE_PCLMUL
        CPU_FEATURE_DTS64
        CPU_FEATURE_MONITOR
        CPU_FEATURE_DS_CPL
        CPU_FEATURE_VMX
        CPU_FEATURE_SMX
        CPU_FEATURE_EST
        CPU_FEATURE_TM2
        CPU_FEATURE_SSSE3
        CPU_FEATURE_CID
        CPU_FEATURE_CX16
        CPU_FEATURE_XTPR
        CPU_FEATURE_PDCM
        CPU_FEATURE_DCA
        CPU_FEATURE_SSE4_1
        CPU_FEATURE_SSE4_2
        CPU_FEATURE_SYSCALL
        CPU_FEATURE_XD
        CPU_FEATURE_MOVBE
        CPU_FEATURE_POPCNT
        CPU_FEATURE_AES
        CPU_FEATURE_XSAVE
        CPU_FEATURE_OSXSAVE
        CPU_FEATURE_AVX
        CPU_FEATURE_MMXEXT
        CPU_FEATURE_3DNOW
        CPU_FEATURE_3DNOWEXT
        CPU_FEATURE_NX
        CPU_FEATURE_FXSR_OPT
        CPU_FEATURE_RDTSCP
        CPU_FEATURE_LM
        CPU_FEATURE_LAHF_LM
        CPU_FEATURE_CMP_LEGACY
        CPU_FEATURE_SVM
        CPU_FEATURE_ABM
        CPU_FEATURE_MISALIGNSSE,
        CPU_FEATURE_SSE4A
        CPU_FEATURE_3DNOWPREFETCH
        CPU_FEATURE_OSVW
        CPU_FEATURE_IBS
        CPU_FEATURE_SSE5
        CPU_FEATURE_SKINIT
        CPU_FEATURE_WDT
        CPU_FEATURE_TS
        CPU_FEATURE_FID
        CPU_FEATURE_VID
        CPU_FEATURE_TTP
        CPU_FEATURE_TM_AMD
        CPU_FEATURE_STC
        CPU_FEATURE_100MHZSTEPS,
        CPU_FEATURE_HWPSTATE
        CPU_FEATURE_CONSTANT_TSC
        CPU_FEATURE_XOP
        CPU_FEATURE_FMA3
        CPU_FEATURE_FMA4
        CPU_FEATURE_TBM
        CPU_FEATURE_F16C
        CPU_FEATURE_RDRAND
        CPU_FEATURE_X2APIC
        CPU_FEATURE_CPB
        CPU_FEATURE_APERFMPERF
        CPU_FEATURE_PFI
        CPU_FEATURE_PA
        CPU_FEATURE_AVX2
        CPU_FEATURE_BMI1
        CPU_FEATURE_BMI2
        CPU_FEATURE_HLE
        CPU_FEATURE_RTM
        CPU_FEATURE_AVX512F
        CPU_FEATURE_AVX512DQ
        CPU_FEATURE_AVX512PF
        CPU_FEATURE_AVX512ER
        CPU_FEATURE_AVX512CD
        CPU_FEATURE_SHA_NI
        CPU_FEATURE_AVX512BW
        CPU_FEATURE_AVX512VL
        CPU_FEATURE_SGX
        CPU_FEATURE_RDSEED
        CPU_FEATURE_ADX
        CPU_FEATURE_AVX512VNNI
        CPU_FEATURE_AVX512VBMI
        CPU_FEATURE_AVX512VBMI2
        NUM_CPU_FEATURES

cdef class CPUID:
    cdef set _features
    cdef _populate_features(self, cpu_id_t cpu_id)
