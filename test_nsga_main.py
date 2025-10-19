import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from nsga_main import (
    plot_final_pareto_comparison,
    export_final_solution_comparison,
    show_parameters,
    run_and_plot_nsga,
)

class TestNSGAMain(unittest.TestCase):

    def setUp(self):
        self.mock_data = {
            "xy_candidate": np.array([[0, 0], [1, 1], [2, 2]]),
            "xy_community": np.array([[0.5, 0.5], [1.5, 1.5]]),
            "Q": np.array([50, 100, 150]),
            "C": np.array([100, 200, 300]),
            "Cp": 10,
            "Cpp": 20,
            "U": 2,
            "D": 1.5,
            "lambda_": 0.1,
            "E_L": np.array([[50], [100]]),
            "E_U": np.array([[60], [120]]),
            "n_candidate": 3,
            "n_community": 2,
            "dpp": np.array([[0, 1.5, 3], [1.5, 0, 2], [3, 2, 0]]),
            "d": np.array([[0.7, 1.7], [0.7, 0.7], [2.1, 0.5]]),
        }
        self.mock_solution = MagicMock()
        self.mock_solution.Cost = [1000, 50]  # Ensure Cost is valid
        self.mock_pareto_front = [self.mock_solution, self.mock_solution]

    @patch("nsga_main.log_with_timestamp")
    def test_show_parameters(self, mock_log_with_timestamp):
        show_parameters(self.mock_data, label="test", log_file="test_log.txt", run_label="test_run")
        mock_log_with_timestamp.assert_called()

    @patch("nsga_main.plt.savefig")
    def test_plot_final_pareto_comparison(self, mock_savefig):
        try:
            plot_final_pareto_comparison(self.mock_pareto_front, self.mock_pareto_front, run_label="test")
            mock_savefig.assert_called_once()
        except ValueError as e:
            self.fail(f"plot_final_pareto_comparison raised ValueError unexpectedly: {e}")

    @patch("nsga_main.pd.DataFrame.to_csv")
    def test_export_final_solution_comparison(self, mock_to_csv):
        try:
            export_final_solution_comparison(self.mock_pareto_front, self.mock_pareto_front, run_label="test")
            mock_to_csv.assert_called()
        except ValueError as e:
            self.fail(f"export_final_solution_comparison raised ValueError unexpectedly: {e}")

    @patch("nsga_main.plt.savefig")
    @patch("nsga_main.log_with_timestamp")
    @patch("nsga_main.nsga_ii_optimization_with_label")
    def test_run_and_plot_nsga(self, mock_nsga_ii, mock_log_with_timestamp, mock_savefig):
        # Mock the return value of nsga_ii_optimization_with_label
        self.mock_solution.x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Open facilities
        self.mock_solution.y = np.array([[50, 0], [0, 100], [0, 0]])  # Demand allocation
        mock_nsga_ii.return_value = (
            self.mock_pareto_front,
            [self.mock_pareto_front],
            {"cost": [[1000, 50]], "stddev": [[50, 10]], "balanced": [[900, 45]]},
        )

        # Call the function
        pareto_front = run_and_plot_nsga(
            self.mock_data,
            label="test",
            MaxIt=10,
            nPop=5,
            pCrossover=0.7,
            pMutation=0.2,
            patience=5,
            return_front=True,
        )

        # Assertions
        self.assertEqual(pareto_front, self.mock_pareto_front)  # Ensure the returned Pareto front is correct
        mock_nsga_ii.assert_called_once()  # Ensure the NSGA-II function was called
        mock_log_with_timestamp.assert_called()  # Ensure logging occurred
        mock_savefig.assert_called()  # Ensure plots were saved