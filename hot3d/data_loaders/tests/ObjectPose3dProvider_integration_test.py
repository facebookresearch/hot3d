# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import os
import tempfile
import unittest

import numpy as np
from data_loaders.constants import POSE_DATA_CSV_COLUMNS
from data_loaders.ObjectPose3dProvider import (
    load_object_pose_trajectory_from_csv,
    load_pose_provider_from_csv,
    ObjectPose3dProvider,
)
from projectaria_tools.core.sensor_data import (  # @manual=fbsource//arvr/projects/ariane/aria_research_kit/projectaria_tools/projectaria_tools:projectaria_tools_package
    TimeDomain,
    TimeQueryOptions,
)


def _make_csv_file(tmp_dir: str, rows: list) -> str:
    """Helper to write a CSV file with POSE_DATA_CSV_COLUMNS header and given rows."""
    filepath = os.path.join(tmp_dir, "dynamic_objects.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(POSE_DATA_CSV_COLUMNS)
        for row in rows:
            writer.writerow(row)
    return filepath


def _make_pose_row(
    object_uid: str,
    timestamp_ns: int,
    tx: float = 1.0,
    ty: float = 2.0,
    tz: float = 3.0,
    qw: float = 1.0,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
) -> list:
    """Create a single CSV row matching POSE_DATA_CSV_COLUMNS order."""
    # POSE_DATA_CSV_COLUMNS order:
    # object_uid, timestamp[ns], t_wo_x[m], t_wo_y[m], t_wo_z[m], q_wo_w, q_wo_x, q_wo_y, q_wo_z
    return [object_uid, timestamp_ns, tx, ty, tz, qw, qx, qy, qz]


class TestObjectPose3dProviderIntegration(unittest.TestCase):
    """Integration tests for ObjectPose3dProvider.

    Each test exercises 2+ real components interacting together.
    """

    def test_csv_loading_produces_valid_provider_with_correct_poses(self) -> None:
        """Components: load_object_pose_trajectory_from_csv + ObjectPose3dProvider + SE3 + ObjectPose3dCollection.

        Verifies that CSV parsing produces a trajectory with correct SE3 poses
        and that the provider correctly exposes timestamps and object UIDs.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _make_pose_row("obj_A", 1000, tx=1.0, ty=2.0, tz=3.0),
                _make_pose_row("obj_B", 1000, tx=4.0, ty=5.0, tz=6.0),
                _make_pose_row("obj_A", 2000, tx=7.0, ty=8.0, tz=9.0),
            ]
            filepath = _make_csv_file(tmp_dir, rows)

            trajectory = load_object_pose_trajectory_from_csv(filepath)
            provider = ObjectPose3dProvider(pose3d_trajectory=trajectory)

            # Verify timestamps are sorted and correct
            self.assertEqual(provider.timestamp_ns_list, [1000, 2000])

            # Verify object UIDs collected across all timestamps
            self.assertEqual(provider.object_uids_with_poses, {"obj_A", "obj_B"})

            # Verify statistics reflect the loaded data
            stats = provider.get_data_statistics()
            self.assertEqual(stats["num_frames"], 2)
            self.assertEqual(stats["num_objects"], 2)
            self.assertIn("obj_A", stats["object_uids"])
            self.assertIn("obj_B", stats["object_uids"])

            # Verify actual SE3 translation values from the CSV
            collection_t1000 = trajectory[1000]
            self.assertIsNotNone(collection_t1000.poses["obj_A"].T_world_object)
            translation_a = np.squeeze(
                collection_t1000.poses["obj_A"].T_world_object.to_quat_and_translation()
            )[4:7]
            np.testing.assert_allclose(translation_a, [1.0, 2.0, 3.0], atol=1e-6)

            translation_b = np.squeeze(
                collection_t1000.poses["obj_B"].T_world_object.to_quat_and_translation()
            )[4:7]
            np.testing.assert_allclose(translation_b, [4.0, 5.0, 6.0], atol=1e-6)

    def test_get_pose_at_timestamp_closest_query(self) -> None:
        """Components: ObjectPose3dProvider + lookup_timestamp (pose_utils) + ObjectPose3dCollectionWithDt.

        Verifies that querying with CLOSEST returns the nearest timestamp's
        collection and correct time delta.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _make_pose_row("cup", 1000),
                _make_pose_row("cup", 3000),
                _make_pose_row("cup", 5000),
            ]
            filepath = _make_csv_file(tmp_dir, rows)
            provider = load_pose_provider_from_csv(filepath)

            # Query between 3000 and 5000, closer to 3000
            result = provider.get_pose_at_timestamp(
                timestamp_ns=3500,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            self.assertIsNotNone(result)
            # Should snap to 3000 (delta = -500 since query - matched = 500)
            self.assertEqual(result.pose3d_collection.timestamp_ns, 3000)
            self.assertEqual(result.time_delta_ns, 500)  # 3500 - 3000

            # Query closer to 5000
            result2 = provider.get_pose_at_timestamp(
                timestamp_ns=4800,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            self.assertIsNotNone(result2)
            self.assertEqual(result2.pose3d_collection.timestamp_ns, 5000)
            self.assertEqual(result2.time_delta_ns, -200)  # 4800 - 5000

    def test_get_pose_at_timestamp_before_query(self) -> None:
        """Components: ObjectPose3dProvider + lookup_timestamp (BEFORE mode) + ObjectPose3dCollectionWithDt.

        Verifies BEFORE query returns the timestamp strictly before the query.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _make_pose_row("plate", 1000),
                _make_pose_row("plate", 3000),
            ]
            filepath = _make_csv_file(tmp_dir, rows)
            provider = load_pose_provider_from_csv(filepath)

            result = provider.get_pose_at_timestamp(
                timestamp_ns=2000,
                time_query_options=TimeQueryOptions.BEFORE,
                time_domain=TimeDomain.TIME_CODE,
            )
            self.assertIsNotNone(result)
            self.assertEqual(result.pose3d_collection.timestamp_ns, 1000)
            self.assertEqual(result.time_delta_ns, 1000)  # 2000 - 1000

    def test_get_pose_at_timestamp_after_query(self) -> None:
        """Components: ObjectPose3dProvider + lookup_timestamp (AFTER mode) + ObjectPose3dCollectionWithDt.

        Verifies AFTER query returns the timestamp strictly after the query.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _make_pose_row("bottle", 1000),
                _make_pose_row("bottle", 3000),
            ]
            filepath = _make_csv_file(tmp_dir, rows)
            provider = load_pose_provider_from_csv(filepath)

            result = provider.get_pose_at_timestamp(
                timestamp_ns=2000,
                time_query_options=TimeQueryOptions.AFTER,
                time_domain=TimeDomain.TIME_CODE,
            )
            self.assertIsNotNone(result)
            self.assertEqual(result.pose3d_collection.timestamp_ns, 3000)
            self.assertEqual(result.time_delta_ns, -1000)  # 2000 - 3000

    def test_acceptable_time_delta_filters_results(self) -> None:
        """Components: ObjectPose3dProvider + lookup_timestamp + acceptable_time_delta filtering.

        Verifies that when the closest match exceeds acceptable_time_delta,
        None is returned.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _make_pose_row("mug", 1000),
                _make_pose_row("mug", 5000),
            ]
            filepath = _make_csv_file(tmp_dir, rows)
            provider = load_pose_provider_from_csv(filepath)

            # Query at 3000, closest is 1000 (delta=2000). With tight delta, should be None.
            result_filtered = provider.get_pose_at_timestamp(
                timestamp_ns=3000,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
                acceptable_time_delta=500,
            )
            self.assertIsNone(result_filtered)

            # Same query with generous delta should succeed
            result_ok = provider.get_pose_at_timestamp(
                timestamp_ns=3000,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
                acceptable_time_delta=5000,
            )
            self.assertIsNotNone(result_ok)
            self.assertIn("mug", result_ok.pose3d_collection.poses)

    def test_exact_timestamp_match_returns_zero_delta(self) -> None:
        """Components: ObjectPose3dProvider + lookup_timestamp + ObjectPose3dCollectionWithDt.

        Verifies that querying an exact timestamp returns time_delta_ns=0.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _make_pose_row("fork", 2000, tx=10.0, ty=20.0, tz=30.0),
            ]
            filepath = _make_csv_file(tmp_dir, rows)
            provider = load_pose_provider_from_csv(filepath)

            result = provider.get_pose_at_timestamp(
                timestamp_ns=2000,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            self.assertIsNotNone(result)
            self.assertEqual(result.time_delta_ns, 0)
            self.assertEqual(result.pose3d_collection.timestamp_ns, 2000)

            # Verify the pose data is correct
            pose = result.pose3d_collection.poses["fork"]
            self.assertIsNotNone(pose.T_world_object)
            translation = np.squeeze(pose.T_world_object.to_quat_and_translation())[4:7]
            np.testing.assert_allclose(translation, [10.0, 20.0, 30.0], atol=1e-6)

    def test_invalid_time_domain_raises_error(self) -> None:
        """Components: ObjectPose3dProvider + get_pose_at_timestamp validation.

        Verifies that using a non-TIME_CODE domain raises ValueError.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [_make_pose_row("obj", 1000)]
            filepath = _make_csv_file(tmp_dir, rows)
            provider = load_pose_provider_from_csv(filepath)

            with self.assertRaises(ValueError):
                provider.get_pose_at_timestamp(
                    timestamp_ns=1000,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.RECORD_TIME,
                )

    def test_load_pose_provider_from_csv_end_to_end(self) -> None:
        """Components: load_pose_provider_from_csv + load_object_pose_trajectory_from_csv + ObjectPose3dProvider + SE3.

        End-to-end test: write CSV, load via convenience function, query poses.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _make_pose_row("knife", 100, tx=0.1, ty=0.2, tz=0.3, qw=1.0),
                _make_pose_row("spoon", 100, tx=0.4, ty=0.5, tz=0.6, qw=1.0),
                _make_pose_row("knife", 200, tx=0.7, ty=0.8, tz=0.9, qw=1.0),
            ]
            filepath = _make_csv_file(tmp_dir, rows)
            provider = load_pose_provider_from_csv(filepath)

            # Verify provider has correct structure
            self.assertEqual(len(provider.timestamp_ns_list), 2)
            self.assertEqual(provider.object_uids_with_poses, {"knife", "spoon"})

            # Query at t=100 and verify both objects present
            result = provider.get_pose_at_timestamp(
                timestamp_ns=100,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            self.assertIsNotNone(result)
            self.assertEqual(len(result.pose3d_collection.poses), 2)
            self.assertIn("knife", result.pose3d_collection.poses)
            self.assertIn("spoon", result.pose3d_collection.poses)

            # Query at t=200 and verify only knife present
            result2 = provider.get_pose_at_timestamp(
                timestamp_ns=200,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            self.assertIsNotNone(result2)
            self.assertEqual(len(result2.pose3d_collection.poses), 1)
            self.assertIn("knife", result2.pose3d_collection.poses)

            # Verify translation values at t=200
            knife_pose = result2.pose3d_collection.poses["knife"]
            translation = np.squeeze(
                knife_pose.T_world_object.to_quat_and_translation()
            )[4:7]
            np.testing.assert_allclose(translation, [0.7, 0.8, 0.9], atol=1e-6)

    def test_multiple_objects_at_multiple_timestamps_with_queries(self) -> None:
        """Components: ObjectPose3dProvider + lookup_timestamp + ObjectPose3dCollection.object_uid_list.

        Tests that the provider correctly handles multiple objects across
        multiple timestamps and that object_uid_list property works correctly
        on each collection.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _make_pose_row("apple", 100),
                _make_pose_row("banana", 100),
                _make_pose_row("cherry", 100),
                _make_pose_row("apple", 300),
                _make_pose_row("banana", 500),
            ]
            filepath = _make_csv_file(tmp_dir, rows)
            provider = load_pose_provider_from_csv(filepath)

            self.assertEqual(provider.timestamp_ns_list, [100, 300, 500])
            self.assertEqual(
                provider.object_uids_with_poses, {"apple", "banana", "cherry"}
            )

            stats = provider.get_data_statistics()
            self.assertEqual(stats["num_frames"], 3)
            self.assertEqual(stats["num_objects"], 3)

            # At t=100, all 3 objects
            result_100 = provider.get_pose_at_timestamp(
                timestamp_ns=100,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            self.assertIsNotNone(result_100)
            self.assertEqual(
                result_100.pose3d_collection.object_uid_list,
                {"apple", "banana", "cherry"},
            )

            # At t=300, only apple
            result_300 = provider.get_pose_at_timestamp(
                timestamp_ns=300,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            self.assertIsNotNone(result_300)
            self.assertEqual(result_300.pose3d_collection.object_uid_list, {"apple"})

    def test_csv_with_non_identity_quaternion(self) -> None:
        """Components: load_object_pose_trajectory_from_csv + SE3.from_quat_and_translation + ObjectPose3dProvider.

        Verifies that non-identity quaternions are correctly parsed and stored
        in the SE3 transform.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 90-degree rotation around Z axis: qw=cos(45)=0.7071, qz=sin(45)=0.7071
            qw = 0.7071067811865476
            qz = 0.7071067811865476
            rows = [
                _make_pose_row(
                    "rotated_obj",
                    500,
                    tx=1.0,
                    ty=2.0,
                    tz=3.0,
                    qw=qw,
                    qx=0.0,
                    qy=0.0,
                    qz=qz,
                ),
            ]
            filepath = _make_csv_file(tmp_dir, rows)
            provider = load_pose_provider_from_csv(filepath)

            result = provider.get_pose_at_timestamp(
                timestamp_ns=500,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            self.assertIsNotNone(result)
            pose = result.pose3d_collection.poses["rotated_obj"]
            quat_and_trans = np.squeeze(pose.T_world_object.to_quat_and_translation())
            # Verify quaternion is preserved (not identity)
            quaternion = quat_and_trans[0:4]  # [qw, qx, qy, qz]
            # The quaternion should have non-zero z component
            self.assertGreater(abs(quaternion[3]), 0.5)  # qz should be ~0.707
            # Verify translation
            np.testing.assert_allclose(quat_and_trans[4:7], [1.0, 2.0, 3.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
