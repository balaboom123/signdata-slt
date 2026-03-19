"""Tests for the component registry system."""

from sign_prep.registry import (
    DATASET_REGISTRY,
    EXTRACTOR_REGISTRY,
    PROCESSOR_REGISTRY,
    register_dataset,
    register_extractor,
    register_processor,
)


class TestRegisterDataset:
    def test_adds_class_to_registry(self):
        @register_dataset("_test_ds")
        class _TestDS:
            pass

        assert "_test_ds" in DATASET_REGISTRY
        assert DATASET_REGISTRY["_test_ds"] is _TestDS

        # Clean up
        del DATASET_REGISTRY["_test_ds"]


class TestRegisterProcessor:
    def test_adds_class_to_registry(self):
        @register_processor("_test_proc")
        class _TestProc:
            pass

        assert "_test_proc" in PROCESSOR_REGISTRY
        assert PROCESSOR_REGISTRY["_test_proc"] is _TestProc

        del PROCESSOR_REGISTRY["_test_proc"]


class TestRegisterExtractor:
    def test_adds_class_to_registry(self):
        @register_extractor("_test_ext")
        class _TestExt:
            pass

        assert "_test_ext" in EXTRACTOR_REGISTRY
        assert EXTRACTOR_REGISTRY["_test_ext"] is _TestExt

        del EXTRACTOR_REGISTRY["_test_ext"]


class TestRealRegistrations:
    """Verify all real registrations exist after importing modules."""

    def test_datasets_registered(self):
        import sign_prep.datasets  # noqa: F401

        assert "youtube_asl" in DATASET_REGISTRY
        assert "how2sign" in DATASET_REGISTRY
        assert len(DATASET_REGISTRY) >= 2

    def test_processors_registered(self):
        import sign_prep.processors  # noqa: F401

        expected = [
            "extract", "normalize", "clip_video",
            "webdataset", "detect_person", "crop_video",
        ]
        for name in expected:
            assert name in PROCESSOR_REGISTRY, f"Processor '{name}' not registered"
        assert len(PROCESSOR_REGISTRY) >= 6

    def test_extractors_registered(self):
        import sign_prep.extractors  # noqa: F401

        assert "mediapipe" in EXTRACTOR_REGISTRY
        assert "mmpose" in EXTRACTOR_REGISTRY
        assert len(EXTRACTOR_REGISTRY) >= 2
