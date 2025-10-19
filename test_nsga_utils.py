import unittest
import os
import numpy as np
import pandas as pd
from nsga_utils import (
    Individual,
    initialize_data,
    log_with_timestamp,
    ensure_output_dir,
    fetch_osm_static_map,
    get_bounding_box,
    plot_all_candidates_with_demand,
    save_cache,
    load_cache,
    cache_exists,
)
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image

class TestNSGAUtils(unittest.TestCase):

    def setUp(self):
        """
        Set up mock data for testing.
        """
        # Mock demand data
        self.Em = np.array([[100], [200], [150]])  # Community demands
        self.xy_community = np.array([[10, 20], [30, 40], [50, 60]])  # Community coordinates

        # Mock candidate data
        self.candidate_file = "mock_candidate_file.xlsx"
        self.xy_candidate = np.array([[5, 15], [25, 35], [45, 55]])
        candidate_df = pd.DataFrame(self.xy_candidate, columns=["x", "y"])
        candidate_df.to_excel(self.candidate_file, index=False)

    def tearDown(self):
        """
        Clean up any temporary files created during testing.
        """
        if os.path.exists(self.candidate_file):
            os.remove(self.candidate_file)
        if os.path.exists("output"):
            for file in os.listdir("output"):
                os.remove(os.path.join("output", file))
            os.rmdir("output")

    def test_initialize_data(self):
        """
        Test the initialize_data function.
        """
        data = initialize_data(self.candidate_file, self.Em, self.xy_community)

        # Check basic properties of the data dictionary
        self.assertEqual(data['n_candidate'], 3)
        self.assertEqual(data['n_community'], 3)
        self.assertIn('Q', data)
        self.assertIn('C', data)
        self.assertIn('dpp', data)
        self.assertIn('gamma', data)

        # Check distance calculations
        self.assertEqual(data['dpp'].shape, (3, 3))  # Distance between candidates
        self.assertEqual(data['d'].shape, (3, 3))  # Distance between candidates and communities

        # Check gamma matrix
        self.assertEqual(data['gamma'].shape, (3, 3))

    def test_log_with_timestamp(self):
        """
        Test the log_with_timestamp function.
        """
        log_file = "test_log.txt"
        log_with_timestamp("Test message", log_file)

        # Check if the log file is created
        self.assertTrue(os.path.exists(log_file))

        # Check the content of the log file
        with open(log_file, "r") as f:
            content = f.read()
        self.assertIn("Test message", content)

        # Clean up
        os.remove(log_file)

    def test_ensure_output_dir(self):
        """
        Test the ensure_output_dir function.
        """
        output_dir = ensure_output_dir()
        self.assertTrue(os.path.exists(output_dir))
        self.assertEqual(output_dir, "output")

    def test_fetch_osm_static_map(self):
        """
        Test the fetch_osm_static_map function.
        """
        min_x, max_x, min_y, max_y = 10, 20, 30, 40

        # Mock the requests.get call
        with patch("nsga_utils.requests.get") as mock_get:
            # Create a mock response object
            mock_response = MagicMock()
            mock_image = Image.new("RGB", (800, 600))  # Create a dummy image
            mock_image_bytes = BytesIO()
            mock_image.save(mock_image_bytes, format="PNG")
            mock_image_bytes.seek(0)
            mock_response.content = mock_image_bytes.getvalue()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            # Call the function
            try:
                img = fetch_osm_static_map(min_x, max_x, min_y, max_y)
                self.assertIsNotNone(img)
                self.assertIsInstance(img, Image.Image)  # Ensure the result is an image
            except Exception as e:
                self.fail(f"fetch_osm_static_map raised an exception: {e}")

    def test_get_bounding_box(self):
        """
        Test the get_bounding_box function.
        """
        min_x, max_x, min_y, max_y = get_bounding_box(self.xy_candidate, self.xy_community)
        self.assertLess(min_x, max_x)
        self.assertLess(min_y, max_y)

    def test_plot_all_candidates_with_demand(self):
        """
        Test the plot_all_candidates_with_demand function.
        """
        ensure_output_dir()
        plot_all_candidates_with_demand(self.xy_candidate, self.xy_candidate, self.xy_community, self.Em)
        plot_path = "output/all_candidates_with_demand.png"
        self.assertTrue(os.path.exists(plot_path))

    def test_cache_functions(self):
        """
        Test the caching functions (save_cache, load_cache, cache_exists).
        """
        cache_file = "test_cache.pkl"
        test_obj = {"key": "value"}

        # Save cache
        save_cache(test_obj, cache_file)
        self.assertTrue(cache_exists(cache_file))

        # Load cache
        loaded_obj = load_cache(cache_file)
        self.assertEqual(test_obj, loaded_obj)

        # Clean up
        os.remove(cache_file)

    def test_individual_class(self):
        """
        Test the Individual class.
        """
        x = np.array([[1, 0], [0, 1]])
        y = np.array([[10, 20], [30, 40]])
        cost = [100, 200]
        individual = Individual(x=x, y=y, Cost=cost)

        self.assertIsNotNone(individual.x)
        self.assertIsNotNone(individual.y)
        if individual.x is not None:
            self.assertTrue(np.array_equal(individual.x, x))
        if individual.y is not None:
            self.assertTrue(np.array_equal(individual.y, y))
        self.assertEqual(individual.Cost, cost)
        self.assertEqual(individual.Rank, None)
        self.assertEqual(individual.DominationSet, [])
        self.assertEqual(individual.DominatedCount, 0)
        self.assertEqual(individual.CrowdingDistance, 0)


if __name__ == "__main__":
    unittest.main()