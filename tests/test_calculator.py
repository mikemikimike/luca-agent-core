import unittest
from unittest.mock import patch

from app.tools.calculator import calculate_metric
from app.tools.registry import registry


class CalculatorTests(unittest.TestCase):
    def test_numeric_arithmetic(self) -> None:
        cases = {
            "2 + 3 * 4": "14",
            "(2 + 3) * 4": "20",
            "-2 ** 2": "-4",
            "2 ** -2": "0.25",
            "8 / 4": "2.0",
            "10 % 3": "1",
            "1.5 + .25": "1.75",
            "+3 - -2": "5",
            " 2 + 3 ": "5",
            "0x10 + 0b10": "18",
            "1_000 + 2": "1002",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(
                    calculate_metric.invoke({"expression": expression}),
                    f"Calculation result: {expected}",
                )

    def test_rejects_non_arithmetic_syntax(self) -> None:
        expressions = (
            "__import__('os')",
            "().__class__",
            "(1).__class__.__base__",
            "(1).__class__.__base__.__subclasses__()",
            "unknown_name",
            "abs(-3)",
            "True",
            "None",
            "'hello'",
            "b'hello'",
            "1j",
            "[]",
            "{}",
            "(1, 2)",
            "[1][0]",
            "[x for x in ()]",
            "lambda: 1",
            "1 < 2",
            "1 and 2",
            "1 if True else 2",
            "1 << 2",
            "1 ^ 2",
            "4 // 2",
            "~1",
            "f'{1}'",
            "x = 1",
            "1; 2",
            "(x := 1)",
            "(1 / 0).__class__",
        )
        for expression in expressions:
            with self.subTest(expression=expression):
                self.assertTrue(
                    calculate_metric.invoke({"expression": expression}).startswith(
                        "Calculation error:"
                    )
                )

    def test_arithmetic_errors_and_recovery(self) -> None:
        for expression in ("1 / 0", "1 % 0", "(", "2 +", "", "   "):
            with self.subTest(expression=expression):
                self.assertTrue(
                    calculate_metric.invoke({"expression": expression}).startswith(
                        "Calculation error:"
                    )
                )
                self.assertEqual(
                    calculate_metric.invoke({"expression": "2 + 3 * 4"}),
                    "Calculation result: 14",
                )

    def test_does_not_use_python_eval(self) -> None:
        with patch(
            "app.tools.calculator.eval",
            side_effect=AssertionError("eval must not run"),
            create=True,
        ):
            self.assertEqual(
                calculate_metric.invoke({"expression": "2 + 3 * 4"}),
                "Calculation result: 14",
            )

    def test_resource_limits_and_real_results(self) -> None:
        expressions = (
            "2 ** 1001",
            "2 ** -1001",
            "9 ** 9 ** 9",
            "(9 ** 9) ** 2",
            "1" * 1001,
            " + ".join(["1"] * 100),
            "1e309",
            "1e308 * 1e308",
            "(-1) ** 0.5",
        )
        for expression in expressions:
            with self.subTest(expression=expression):
                self.assertTrue(
                    calculate_metric.invoke({"expression": expression}).startswith(
                        "Calculation error:"
                    )
                )
        self.assertEqual(
            calculate_metric.invoke({"expression": "2 ** 32"}),
            "Calculation result: 4294967296",
        )

    def test_limit_boundaries(self) -> None:
        for expression, expected in (
            ("1" + " " * 999, "1"),
            (" + ".join(["1"] * 43), "43"),
            ("2 ** 1000", str(2**1000)),
            ("2 ** -1000", str(2**-1000)),
        ):
            with self.subTest(expression=expression):
                self.assertEqual(
                    calculate_metric.invoke({"expression": expression}),
                    f"Calculation result: {expected}",
                )
        for expression in (" + ".join(["1"] * 44), "-" * 200 + "1"):
            with self.subTest(expression=expression):
                self.assertTrue(
                    calculate_metric.invoke({"expression": expression}).startswith(
                        "Calculation error:"
                    )
                )

    def test_whole_tree_is_validated_before_evaluation(self) -> None:
        with patch("app.tools.calculator._bounded_power") as power:
            self.assertTrue(
                calculate_metric.invoke(
                    {"expression": "(2 ** 8) + (1).__class__"}
                ).startswith("Calculation error:")
            )
            power.assert_not_called()

    def test_registry_retains_public_tool_schema(self) -> None:
        self.assertIs(registry.get_tool("calculate_metric"), calculate_metric)
        manifest = next(
            item
            for item in registry.get_manifest()
            if item["name"] == "calculate_metric"
        )
        self.assertEqual(manifest["parameters"]["required"], ["expression"])
        self.assertEqual(
            manifest["parameters"]["properties"]["expression"]["type"], "string"
        )


if __name__ == "__main__":
    unittest.main()
