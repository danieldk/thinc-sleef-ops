cdef class CPUID:
    def __init__(self):
        self._features = set()

        if cpuid_present() == 0:
            return

        cdef cpu_raw_data_t cpu_raw_data
        cdef cpu_id_t cpu_id

        if cpuid_get_raw_data(&cpu_raw_data) < 0:
            return

        if cpu_identify(&cpu_raw_data, &cpu_id) < 0:
            return

        self._populate_features(cpu_id)

    cdef _populate_features(self, cpu_id_t cpu_id):
        features = set()
        cdef cpu_feature_t num_features = NUM_CPU_FEATURES
        cdef cpu_feature_t feature
        for feature in range(num_features):
            if cpu_id.flags[int(feature)] == 1:
                features.add(cpu_feature_str(feature).decode("ascii"))
        self._features = features

    @property
    def features(self):
        return set(self._features)
