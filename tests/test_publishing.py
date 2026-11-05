from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from souqlens.publishing import publish_space, wait_for_space


class FakeHub:
    def __init__(self, sdk="gradio", hardware="zero-a10g", requested=None):
        self.sdk, self.hardware, self.requested = sdk, hardware, requested
        self.uploads = []

    def space_info(self, repo_id):
        return SimpleNamespace(sdk=self.sdk, sha="current-space-commit")

    def get_space_runtime(self, repo_id):
        return SimpleNamespace(hardware=self.hardware, requested_hardware=self.requested)

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)
        return SimpleNamespace(oid="uploaded-space-commit")


class PublishingTests(unittest.TestCase):
    def runtime_api(self, stages, commit="expected"):
        class RuntimeHub:
            def __init__(self):
                self.stages=iter(stages)
            def space_info(self, repo_id):
                return SimpleNamespace(sha=commit)
            def get_space_runtime(self, repo_id):
                return SimpleNamespace(stage=next(self.stages))
        return RuntimeHub()

    def test_runtime_verification_waits_for_running(self):
        ticks=[0]
        result=wait_for_space("Hussam07/souqlens","expected",api=self.runtime_api(["BUILDING","STARTING","RUNNING"]),
            clock=lambda:ticks[0],sleeper=lambda seconds:ticks.__setitem__(0,ticks[0]+seconds))
        self.assertEqual(result["status"],"running")
        self.assertEqual(ticks[0],20)

    def test_runtime_failure_and_changed_source_are_reported(self):
        with self.assertRaisesRegex(RuntimeError,"RUNTIME_ERROR"):
            wait_for_space("Hussam07/souqlens","expected",api=self.runtime_api(["RUNTIME_ERROR"]))
        with self.assertRaisesRegex(RuntimeError,"source changed"):
            wait_for_space("Hussam07/souqlens","expected",api=self.runtime_api(["RUNNING"],"other"))

    def test_runtime_timeout_is_bounded(self):
        ticks=[0]
        with self.assertRaisesRegex(RuntimeError,"last stage: BUILDING"):
            wait_for_space("Hussam07/souqlens","expected",timeout=5,interval=10,
                api=self.runtime_api(["BUILDING","BUILDING"]),clock=lambda:ticks[0],
                sleeper=lambda seconds:ticks.__setitem__(0,ticks[0]+seconds))
        self.assertEqual(ticks[0],5)

    def stage(self, root):
        for name in ["app.py", "README.md", "requirements.txt", "pyproject.toml", "src/souqlens/api.py"]:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text("sdk: gradio" if name=="README.md" else "fixture")

    def test_uploads_only_to_existing_gradio_zerogpu_with_parent_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.stage(root);api=FakeHub()
            result=publish_space(root,"Hussam07/souqlens","abc1234",api)
            self.assertEqual(result["status"],"source_uploaded")
            self.assertEqual(api.uploads[0]["parent_commit"],"current-space-commit")
            self.assertEqual(api.uploads[0]["repo_type"],"space")

    def test_wrong_sdk_stops_before_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.stage(root);api=FakeHub(sdk="docker")
            with self.assertRaisesRegex(ValueError,"Gradio"):
                publish_space(root,"Hussam07/souqlens",api=api)
            self.assertEqual(api.uploads,[])

    def test_nonzero_or_unknown_hardware_stops_before_upload(self):
        for hardware,requested in [("cpu-basic",None),(None,None),("t4-small",None),("zero-a10g","t4-small")]:
            with self.subTest(hardware=hardware),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);self.stage(root);api=FakeHub(hardware=hardware,requested=requested)
                with self.assertRaisesRegex(ValueError,"ZeroGPU"):
                    publish_space(root,"Hussam07/souqlens",api=api)
                self.assertEqual(api.uploads,[])

    def test_unexpected_data_file_is_rejected_before_contacting_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.stage(root);api=FakeHub()
            (root/"catalog.csv").write_text("private product data")
            with self.assertRaisesRegex(ValueError,"Unexpected file"):
                publish_space(root,"Hussam07/souqlens",api=api)
            self.assertEqual(api.uploads,[])
