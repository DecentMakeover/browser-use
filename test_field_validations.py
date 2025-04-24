#!/usr/bin/env python3
import sys
import csv
from validation import validate_pan, validate_examiner_prefix, validate_remitter_prefix, validate_remitter_name, validate_beneficiary_prefix, validate_beneficiary_name, validate_accountant_membership_no, validate_accountant_registration_no

def run_validation_tests(test_name, test_cases, validation_function):
    """
    Generic function to run validation tests for any field.
    
    Args:
        test_name: Name of the test (e.g., "PAN", "Examiner Prefix")
        test_cases: List of test case dictionaries
        validation_function: The validation function to test
        
    Returns:
        tuple: (pass_count, total_tests)
    """
    # Run tests and collect results
    results = []
    
    for test_case in test_cases:
        value = test_case["value"]
        
        try:
            is_valid, error_msg = validation_function(value)
        except Exception as e:
            # Handle any exceptions during validation
            is_valid = False
            error_msg = str(e)
        
        # Determine if test passed or failed based on expected outcome
        test_passed = (is_valid == test_case["should_pass"])
        if not test_passed and test_case["should_pass"]:
            status = "FAIL (Expected valid, but was rejected)"
        elif not test_passed and not test_case["should_pass"]:
            status = "FAIL (Expected invalid, but was accepted)"
        elif is_valid:
            status = f"PASS (Valid {test_name})"
        else:
            status = f"PASS (Invalid {test_name})"
        
        results.append({
            "id": test_case["id"],
            "value": value,
            "reason": test_case["reason"],
            "is_valid": is_valid,
            "error_msg": error_msg,
            "should_pass": test_case["should_pass"],
            "test_passed": test_passed,
            "status": status
        })
    
    # Print console report
    print("\n" + "="*100)
    print(f" {test_name.upper()} VALIDATION TEST REPORT ".center(100, "="))
    print("="*100)
    
    print(f"{'ID':^5} | {'Value':^14} | {'Valid':^7} | {'Test':^6} | {'Reason/Error':<60}")
    print("-"*100)
    
    for result in results:
        value_display = repr(result['value']) if result['value'] is not None else 'None'
        if len(value_display) > 14:
            value_display = value_display[:11] + '...'
        print(f"{result['id']:5d} | {value_display:14} | {str(result['is_valid']):^7} | {'✓' if result['test_passed'] else '✗':^6} | {result['error_msg'] if not result['is_valid'] else result['reason']:.60}")
    
    # Summary stats
    total_tests = len(results)
    pass_count = sum(1 for r in results if r['test_passed'])
    fail_count = total_tests - pass_count
    
    print("-"*100)
    print(f"SUMMARY: {pass_count}/{total_tests} tests passed ({pass_count/total_tests*100:.1f}%)")
    print("="*100)
    
    # Generate CSV report
    report_filename = f"{test_name.lower().replace(' ', '_')}_validation_report.csv"
    with open(report_filename, 'w', newline='') as csvfile:
        fieldnames = ['Test ID', 'Value', 'Reason', 'Is Valid', 'Error Message', 'Should Pass', 'Test Passed', 'Status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'Test ID': result['id'],
                'Value': repr(result['value']),
                'Reason': result['reason'],
                'Is Valid': result['is_valid'],
                'Error Message': result['error_msg'],
                'Should Pass': result['should_pass'],
                'Test Passed': result['test_passed'],
                'Status': result['status']
            })
    
    print(f"Detailed report saved to {report_filename}")
    
    return pass_count, total_tests

def get_pan_test_cases():
    """Return test cases for PAN validation."""
    # Original test cases
    original_test_cases = [
        {"id": 1, "value": "aaapz1234c", "reason": "All lowercase (should be uppercase)", "should_pass": True},
        {"id": 2, "value": "AAAPZ1234", "reason": "Only 9 characters (missing last letter)", "should_pass": False},
        {"id": 3, "value": "AAAPZ1234CC", "reason": "11 characters (one extra character)", "should_pass": False},
        {"id": 4, "value": "A1APZ1234C", "reason": "First 5 characters must be letters only", "should_pass": False},
        {"id": 5, "value": "AAAP11234C", "reason": "Letters replaced with numbers in the first five", "should_pass": False},
        {"id": 6, "value": "AAAAZ1234C", "reason": "Fourth character should be a valid holder type (P, C, etc.)", "should_pass": False},
        {"id": 7, "value": "AAAPZ12C4C", "reason": "Digits not in correct positions (positions 6–9 should be digits)", "should_pass": False},
        {"id": 8, "value": "AAAPZ12#4C", "reason": "Special character used (not allowed)", "should_pass": False},
        {"id": 9, "value": "1234567890", "reason": "All digits, not alphanumeric format", "should_pass": False},
        {"id": 10, "value": "AAAAZ1234X", "reason": "Fifth character doesn't match first letter of surname/name", "should_pass": False},
        {"id": 11, "value": "AAAPZ123", "reason": "Only 8 characters (incomplete)", "should_pass": False},
        {"id": 12, "value": "A@APZ1234C", "reason": "Special character @ in the first five letters", "should_pass": False},
        {"id": 13, "value": "A1APZ12C4!", "reason": "Multiple issues: numbers in place of letters and invalid characters", "should_pass": False},
        {"id": 14, "value": "ZZZZZ9999Z", "reason": "Technically valid format, but all placeholder-type values", "should_pass": True},
        {"id": 15, "value": "ABCPZ1234C", "reason": "Valid PAN", "should_pass": True},
        {"id": 16, "value": "AAAPZ1234C", "reason": "Valid PAN", "should_pass": True},
    ]
    
    # New test cases
    new_test_cases = [
        {"id": 101, "value": "", "reason": "Empty string", "should_pass": False},
        {"id": 102, "value": " ", "reason": "Only whitespace", "should_pass": False},
        {"id": 103, "value": "A1C", "reason": "Too short (< 10 chars)", "should_pass": False},
        {"id": 104, "value": "AAAAPZ1234C", "reason": "Too long (11 chars)", "should_pass": False},
        {"id": 105, "value": "AA APZ1234C", "reason": "Contains space inside", "should_pass": False},
        {"id": 106, "value": "AA@PZ1234C", "reason": "Special character in first 5 slots", "should_pass": False},
        {"id": 107, "value": "AAAPZ1234C\n", "reason": "Trailing newline / hidden control char", "should_pass": False},
        {"id": 108, "value": "aAAPZ1234C", "reason": "Lower‑case letter among mandatory upper case", "should_pass": True},  # Passes due to auto-uppercase
        {"id": 109, "value": "AAAPz1234C", "reason": "Lower‑case 5th char", "should_pass": True},  # Passes due to auto-uppercase
        {"id": 110, "value": "AAA4Z1234C", "reason": "Digit where letter expected (1st–5th pos)", "should_pass": False},
        {"id": 111, "value": "AAAAZ12A4C", "reason": "Letter inside digit zone (6th–9th pos)", "should_pass": False},
        {"id": 112, "value": "AAAAP1234C", "reason": "4th char P ok, but 5th char must be letter of surname – here digit", "should_pass": False},
        {"id": 113, "value": "AAAA01234C", "reason": "4th char 0 (invalid category code)", "should_pass": False},
        {"id": 114, "value": "AAAAZ1234@", "reason": "10th char not a letter", "should_pass": False},
        {"id": 115, "value": "AAAAZ12345", "reason": "10th char digit, not letter", "should_pass": False},
        {"id": 116, "value": "1234Z1234C", "reason": "Starts with digits, should be letters", "should_pass": False},
        {"id": 117, "value": "!!!!!1234C", "reason": "Non‑alphanumerics in letter zone", "should_pass": False},
        {"id": 118, "value": "AAAAI1234C", "reason": "4th char I – not in allowed set {A B C F G H L J P T}", "should_pass": False},
        {"id": 119, "value": "AAAAÜ1234C", "reason": "Non‑ASCII letter in first 5 slots", "should_pass": False},
        {"id": 120, "value": "AAAAP12C", "reason": "Missing one digit in numeric zone (only 3 digits)", "should_pass": False},
        {"id": 121, "value": "AAAAP12345C", "reason": "One extra digit in numeric zone (5 digits)", "should_pass": False},
        {"id": 122, "value": "  AAAAP1234C", "reason": "Leading space", "should_pass": False},
        {"id": 123, "value": "AAAAP1234C ", "reason": "Trailing space", "should_pass": False},
        {"id": 124, "value": " AAAAP1234C", "reason": "Leading non‑breaking space (Unicode U+00A0)", "should_pass": False},
        {"id": 125, "value": "AAAAP１２３４C", "reason": "Digits are full‑width Unicode, not 0–9 ASCII", "should_pass": False},
        {"id": 126, "value": "AAAAP1234Ç", "reason": "10th char non‑English letter with diacritic", "should_pass": False},
        {"id": 127, "value": "AAAAP1234c", "reason": "10th char lowercase", "should_pass": False},  # Should fail despite auto-uppercase
        {"id": 128, "value": "AAAAP1O34C", "reason": "Letter O used instead of zero in numeric zone", "should_pass": False},
        {"id": 129, "value": "AAAAP123IC", "reason": "Letter I inside numeric zone", "should_pass": False},
    ]
    
    # Return combined test cases
    return original_test_cases + new_test_cases

def get_examiner_prefix_test_cases():
    """Return test cases for Examiner Prefix validation."""
    return [
        {"id": 1, "value": "I", "reason": "Valid value 'I'", "should_pass": True},
        {"id": 2, "value": "We", "reason": "Valid value 'We'", "should_pass": True},
        {"id": 3, "value": "i", "reason": "Lowercase 'i' (should be normalized to 'I')", "should_pass": True},
        {"id": 4, "value": "we", "reason": "Lowercase 'we' (should be normalized to 'We')", "should_pass": True},
        {"id": 5, "value": "WE", "reason": "Uppercase 'WE' (should be normalized to 'We')", "should_pass": True},
        {"id": 6, "value": "", "reason": "Empty string", "should_pass": False},
        {"id": 7, "value": " ", "reason": "Only whitespace", "should_pass": False},
        {"id": 8, "value": " I", "reason": "Leading whitespace (valid but should be normalized)", "should_pass": True},
        {"id": 9, "value": "I ", "reason": "Trailing whitespace (valid but should be normalized)", "should_pass": True},
        {"id": 10, "value": " We ", "reason": "Leading and trailing whitespace (valid but should be normalized)", "should_pass": True},
        {"id": 11, "value": "I/We", "reason": "Invalid value 'I/We'", "should_pass": False},
        {"id": 12, "value": "Me", "reason": "Invalid value 'Me'", "should_pass": False},
        {"id": 13, "value": "Us", "reason": "Invalid value 'Us'", "should_pass": False},
        {"id": 14, "value": "They", "reason": "Invalid value 'They'", "should_pass": False},
        {"id": 15, "value": "1", "reason": "Numeric value", "should_pass": False},
        {"id": 16, "value": "I\n", "reason": "Contains newline (valid but should be normalized)", "should_pass": True},
        {"id": 17, "value": None, "reason": "None value", "should_pass": False},
    ]

def get_remitter_prefix_test_cases():
    """Return test cases for Remitter Prefix validation."""
    return [
        {"id": 1, "value": "Mr", "reason": "Valid value 'Mr'", "should_pass": True},
        {"id": 2, "value": "Ms", "reason": "Valid value 'Ms'", "should_pass": True},
        {"id": 3, "value": "M/s", "reason": "Valid value 'M/s'", "should_pass": True},
        {"id": 4, "value": "mr", "reason": "Lowercase 'mr' (should be normalized to 'Mr')", "should_pass": True},
        {"id": 5, "value": "ms", "reason": "Lowercase 'ms' (should be normalized to 'Ms')", "should_pass": True},
        {"id": 6, "value": "m/s", "reason": "Lowercase 'm/s' (should be normalized to 'M/s')", "should_pass": True},
        {"id": 7, "value": "MR", "reason": "Uppercase 'MR' (should be normalized to 'Mr')", "should_pass": True},
        {"id": 8, "value": "MS", "reason": "Uppercase 'MS' (should be normalized to 'Ms')", "should_pass": True},
        {"id": 9, "value": "M/S", "reason": "Uppercase 'M/S' (should be normalized to 'M/s')", "should_pass": True},
        {"id": 10, "value": " Mr", "reason": "Leading whitespace (valid but should be normalized)", "should_pass": True},
        {"id": 11, "value": "Ms ", "reason": "Trailing whitespace (valid but should be normalized)", "should_pass": True},
        {"id": 12, "value": " M/s ", "reason": "Leading and trailing whitespace (valid but should be normalized)", "should_pass": True},
        {"id": 13, "value": "Mr\n", "reason": "Contains newline (valid but should be normalized)", "should_pass": True},
        {"id": 14, "value": "", "reason": "Empty string", "should_pass": False},
        {"id": 15, "value": " ", "reason": "Only whitespace", "should_pass": False},
        {"id": 16, "value": "M", "reason": "Invalid value 'M'", "should_pass": False},
        {"id": 17, "value": "Mister", "reason": "Invalid value 'Mister'", "should_pass": False},
        {"id": 18, "value": "Miss", "reason": "Invalid value 'Miss'", "should_pass": False},
        {"id": 19, "value": "Mrs", "reason": "Invalid value 'Mrs'", "should_pass": False},
        {"id": 20, "value": "Mr.", "reason": "Invalid value with period 'Mr.'", "should_pass": False},
        {"id": 21, "value": "M\\s", "reason": "Invalid value with backslash instead of forward slash", "should_pass": False},
        {"id": 22, "value": None, "reason": "None value", "should_pass": False},
    ]

def get_remitter_name_test_cases():
    """Return test cases for Remitter Name validation."""
    return [
        {"id": 1, "value": "John Smith", "reason": "Valid simple name", "should_pass": True},
        {"id": 2, "value": "Jane Doe", "reason": "Valid simple name", "should_pass": True},
        {"id": 3, "value": "Robert O'Connor", "reason": "Valid name with apostrophe", "should_pass": True},
        {"id": 4, "value": "Mary-Jane Wilson", "reason": "Valid name with hyphen", "should_pass": True},
        {"id": 5, "value": "Dr. James Brown", "reason": "Valid name with title and period", "should_pass": True},
        {"id": 6, "value": "José Ángel García", "reason": "Valid name with accented characters", "should_pass": True},
        {"id": 7, "value": "Anna Maria Louisa Italiano", "reason": "Valid longer name", "should_pass": True},
        {"id": 8, "value": " John Smith", "reason": "Valid name with leading whitespace", "should_pass": True},
        {"id": 9, "value": "Jane Doe ", "reason": "Valid name with trailing whitespace", "should_pass": True},
        {"id": 10, "value": " Robert Johnson ", "reason": "Valid name with leading and trailing whitespace", "should_pass": True},
        {"id": 11, "value": "", "reason": "Empty string", "should_pass": False},
        {"id": 12, "value": " ", "reason": "Only whitespace", "should_pass": False},
        {"id": 13, "value": "J", "reason": "Too short (less than 2 characters)", "should_pass": False},
        {"id": 14, "value": "John123", "reason": "Contains numbers", "should_pass": False},
        {"id": 15, "value": "John@Smith", "reason": "Contains special character @", "should_pass": False},
        {"id": 16, "value": "John  Smith", "reason": "Contains consecutive spaces", "should_pass": False},
        {"id": 17, "value": "John--Smith", "reason": "Contains consecutive hyphens", "should_pass": False},
        {"id": 18, "value": "John..Smith", "reason": "Contains consecutive periods", "should_pass": False},
        {"id": 19, "value": "John''Smith", "reason": "Contains consecutive apostrophes", "should_pass": False},
        {"id": 20, "value": ".", "reason": "Only a special character, no letters", "should_pass": False},
        {"id": 21, "value": "123", "reason": "Only numbers, no letters", "should_pass": False},
        {"id": 22, "value": None, "reason": "None value", "should_pass": False},
        {"id": 23, "value": "A" * 101, "reason": "Too long (over 100 characters)", "should_pass": False},
    ]

def get_beneficiary_prefix_test_cases():
    """Return test cases for Beneficiary Prefix validation."""
    return [
        {"id": 1, "value": "Mr", "reason": "Valid value 'Mr'", "should_pass": True},
        {"id": 2, "value": "Ms", "reason": "Valid value 'Ms'", "should_pass": True},
        {"id": 3, "value": "M/s", "reason": "Valid value 'M/s'", "should_pass": True},
        {"id": 4, "value": "mr", "reason": "Lowercase 'mr' (should be normalized to 'Mr')", "should_pass": True},
        {"id": 5, "value": "ms", "reason": "Lowercase 'ms' (should be normalized to 'Ms')", "should_pass": True},
        {"id": 6, "value": "m/s", "reason": "Lowercase 'm/s' (should be normalized to 'M/s')", "should_pass": True},
        {"id": 7, "value": "MR", "reason": "Uppercase 'MR' (should be normalized to 'Mr')", "should_pass": True},
        {"id": 8, "value": "MS", "reason": "Uppercase 'MS' (should be normalized to 'Ms')", "should_pass": True},
        {"id": 9, "value": "M/S", "reason": "Uppercase 'M/S' (should be normalized to 'M/s')", "should_pass": True},
        {"id": 10, "value": " Mr", "reason": "Leading whitespace (valid but should be normalized)", "should_pass": True},
        {"id": 11, "value": "Ms ", "reason": "Trailing whitespace (valid but should be normalized)", "should_pass": True},
        {"id": 12, "value": " M/s ", "reason": "Leading and trailing whitespace (valid but should be normalized)", "should_pass": True},
        {"id": 13, "value": "Mr\n", "reason": "Contains newline (valid but should be normalized)", "should_pass": True},
        {"id": 14, "value": "", "reason": "Empty string", "should_pass": False},
        {"id": 15, "value": " ", "reason": "Only whitespace", "should_pass": False},
        {"id": 16, "value": "M", "reason": "Invalid value 'M'", "should_pass": False},
        {"id": 17, "value": "Mister", "reason": "Invalid value 'Mister'", "should_pass": False},
        {"id": 18, "value": "Miss", "reason": "Invalid value 'Miss'", "should_pass": False},
        {"id": 19, "value": "Mrs", "reason": "Invalid value 'Mrs'", "should_pass": False},
        {"id": 20, "value": "Mr.", "reason": "Invalid value with period 'Mr.'", "should_pass": False},
        {"id": 21, "value": "M\\s", "reason": "Invalid value with backslash instead of forward slash", "should_pass": False},
        {"id": 22, "value": None, "reason": "None value", "should_pass": False},
    ]

def get_beneficiary_name_test_cases():
    """Return test cases for Beneficiary Name validation."""
    return [
        {"id": 1, "value": "John Smith", "reason": "Valid simple name", "should_pass": True},
        {"id": 2, "value": "Jane Doe", "reason": "Valid simple name", "should_pass": True},
        {"id": 3, "value": "Robert O'Connor", "reason": "Valid name with apostrophe", "should_pass": True},
        {"id": 4, "value": "Mary-Jane Wilson", "reason": "Valid name with hyphen", "should_pass": True},
        {"id": 5, "value": "Dr. James Brown", "reason": "Valid name with title and period", "should_pass": True},
        {"id": 6, "value": "José Ángel García", "reason": "Valid name with accented characters", "should_pass": True},
        {"id": 7, "value": "Anna Maria Louisa Italiano", "reason": "Valid longer name", "should_pass": True},
        {"id": 8, "value": " John Smith", "reason": "Valid name with leading whitespace", "should_pass": True},
        {"id": 9, "value": "Jane Doe ", "reason": "Valid name with trailing whitespace", "should_pass": True},
        {"id": 10, "value": " Robert Johnson ", "reason": "Valid name with leading and trailing whitespace", "should_pass": True},
        {"id": 11, "value": "", "reason": "Empty string", "should_pass": False},
        {"id": 12, "value": " ", "reason": "Only whitespace", "should_pass": False},
        {"id": 13, "value": "J", "reason": "Too short (less than 2 characters)", "should_pass": False},
        {"id": 14, "value": "John123", "reason": "Contains numbers", "should_pass": False},
        {"id": 15, "value": "John@Smith", "reason": "Contains special character @", "should_pass": False},
        {"id": 16, "value": "John  Smith", "reason": "Contains consecutive spaces", "should_pass": False},
        {"id": 17, "value": "John--Smith", "reason": "Contains consecutive hyphens", "should_pass": False},
        {"id": 18, "value": "John..Smith", "reason": "Contains consecutive periods", "should_pass": False},
        {"id": 19, "value": "John''Smith", "reason": "Contains consecutive apostrophes", "should_pass": False},
        {"id": 20, "value": ".", "reason": "Only a special character, no letters", "should_pass": False},
        {"id": 21, "value": "123", "reason": "Only numbers, no letters", "should_pass": False},
        {"id": 22, "value": None, "reason": "None value", "should_pass": False},
        {"id": 23, "value": "A" * 101, "reason": "Too long (over 100 characters)", "should_pass": False},
    ]

def get_accountant_membership_no_test_cases():
    """Return test cases for Accountant Membership No validation."""
    return [
        {"id": 1, "value": "123456", "reason": "Valid numeric membership number", "should_pass": True},
        {"id": 2, "value": "9876543210", "reason": "Valid longer numeric membership number", "should_pass": True},
        {"id": 3, "value": "0", "reason": "Valid single digit membership number", "should_pass": True},
        {"id": 4, "value": " 123456 ", "reason": "Valid with whitespace (should be normalized)", "should_pass": True},
        {"id": 5, "value": "", "reason": "Empty string", "should_pass": False},
        {"id": 6, "value": " ", "reason": "Only whitespace", "should_pass": False},
        {"id": 7, "value": "ABC123", "reason": "Contains letters", "should_pass": False},
        {"id": 8, "value": "123-456", "reason": "Contains hyphens", "should_pass": False},
        {"id": 9, "value": "123.456", "reason": "Contains periods", "should_pass": False},
        {"id": 10, "value": "123 456", "reason": "Contains spaces between digits", "should_pass": False},
        {"id": 11, "value": "123@456", "reason": "Contains special characters", "should_pass": False},
        {"id": 12, "value": None, "reason": "None value", "should_pass": False},
    ]

def get_accountant_registration_no_test_cases():
    """Return test cases for Accountant Registration No validation."""
    return [
        {"id": 1, "value": "123456", "reason": "Valid numeric registration number", "should_pass": True},
        {"id": 2, "value": "9876543210", "reason": "Valid longer numeric registration number", "should_pass": True},
        {"id": 3, "value": "0", "reason": "Valid single digit registration number", "should_pass": True},
        {"id": 4, "value": " 123456 ", "reason": "Valid with whitespace (should be normalized)", "should_pass": True},
        {"id": 5, "value": "", "reason": "Empty string", "should_pass": False},
        {"id": 6, "value": " ", "reason": "Only whitespace", "should_pass": False},
        {"id": 7, "value": "ABC123", "reason": "Contains letters", "should_pass": False},
        {"id": 8, "value": "123-456", "reason": "Contains hyphens", "should_pass": False},
        {"id": 9, "value": "123.456", "reason": "Contains periods", "should_pass": False},
        {"id": 10, "value": "123 456", "reason": "Contains spaces between digits", "should_pass": False},
        {"id": 11, "value": "123@456", "reason": "Contains special characters", "should_pass": False},
        {"id": 12, "value": None, "reason": "None value", "should_pass": False},
    ]

def main():
    """Run validation tests for multiple fields."""
    tests = [
        {"name": "PAN", "test_cases": get_pan_test_cases(), "validation_function": validate_pan},
        {"name": "Examiner Prefix", "test_cases": get_examiner_prefix_test_cases(), "validation_function": validate_examiner_prefix},
        {"name": "Remitter Prefix", "test_cases": get_remitter_prefix_test_cases(), "validation_function": validate_remitter_prefix},
        {"name": "Remitter Name", "test_cases": get_remitter_name_test_cases(), "validation_function": validate_remitter_name},
        {"name": "Beneficiary Prefix", "test_cases": get_beneficiary_prefix_test_cases(), "validation_function": validate_beneficiary_prefix},
        {"name": "Beneficiary Name", "test_cases": get_beneficiary_name_test_cases(), "validation_function": validate_beneficiary_name},
        {"name": "Accountant Membership No", "test_cases": get_accountant_membership_no_test_cases(), "validation_function": validate_accountant_membership_no},
        {"name": "Accountant Registration No", "test_cases": get_accountant_registration_no_test_cases(), "validation_function": validate_accountant_registration_no},
        # Add more field tests here as needed
    ]
    
    all_results = []
    
    for test in tests:
        print(f"\nRunning {test['name']} validation tests...")
        pass_count, total = run_validation_tests(test["name"], test["test_cases"], test["validation_function"])
        all_results.append((test["name"], pass_count, total))
    
    # Print overall summary
    print("\n" + "="*100)
    print(" OVERALL VALIDATION TEST RESULTS ".center(100, "="))
    print("="*100)
    
    total_passed = 0
    total_tests = 0
    
    for name, passes, total in all_results:
        print(f"{name:20} | {passes:4d}/{total:4d} passed | {passes/total*100:6.1f}%")
        total_passed += passes
        total_tests += total
    
    print("-"*100)
    print(f"TOTAL:            | {total_passed:4d}/{total_tests:4d} passed | {total_passed/total_tests*100:6.1f}%")
    print("="*100)
    
    # Return exit code based on test success
    if total_passed == total_tests:
        return 0  # All tests passed
    else:
        return 1  # Some tests failed

if __name__ == "__main__":
    sys.exit(main()) 