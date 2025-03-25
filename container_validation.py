import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants for validation
CONTAINER_LENGTH = 11

# OCR misread mappings
OCR_NUMBER_TO_LETTER = {
    '1': 'I', '7': 'I', '4': 'A', '6': 'G', '8': 'B', '0': 'O',
    '5': 'S', '2': 'Z'
}

OCR_LETTER_TO_NUMBER = {
    'I': '1', 'L': '1', 'A': '4', 'G': '6', 'B': '8', 'K': '5',
    'C': '5', 'O': '0', 'S': '5', 'Z': '2'
}

# Default carrier prefixes
DEFAULT_PREFIXES = {"APHU", "EGHU", "OOLU", "ADMU"}  # Add more as needed

# Cache for carrier prefixes
_cached_prefixes = None


def calculate_iso_checksum(container_number: str) -> int:
    """
    Calculate the checksum for the container number based on ISO 6346.
    :param container_number: The container number (first 10 characters).
    :return: The calculated check digit.
    :raises ValueError: If the container number is not 10 characters long or contains invalid characters.
    """
    if len(container_number) != 10:
        raise ValueError("Container number must be exactly 10 characters long.")

    # Mapping of letters to their corresponding values
    letter_values = {
        'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20,
        'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31,
        'U': 32, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38
    }

    digits = []
    for char in container_number:
        if char.isdigit():  # Digit remains the same
            digits.append(int(char))
        elif char.isalpha():  # Letter is converted to its value
            char_upper = char.upper()
            if char_upper in letter_values:
                digits.append(letter_values[char_upper])
            else:
                logging.debug(f"Invalid character in container number: {char}. Skipping.")
                digits.append(0)  # Default to 0 for invalid letters
        else:
            logging.debug(f"Invalid character in container number: {char}. Skipping.")
            digits.append(0)  # Skip invalid characters

    # Calculate weighted sum
    weighted_sum = sum(value * (2 ** i) for i, value in enumerate(digits))
    checksum = weighted_sum % 11
    return 0 if checksum == 10 else checksum


def get_carrier_prefixes() -> tuple:
    """
    Read the pre-defined container prefixes from a file and return them as a tuple.
    :return: A tuple of valid carrier prefixes, e.g., ("APHU", "EGHU", ...).
    """
    global _cached_prefixes
    if _cached_prefixes is not None:
        return _cached_prefixes

    try:
        with open("./container_prefix.txt", "r") as f:
            lines = f.readlines()
            if not lines:
                logging.warning("No prefixes found in the file. Using default prefixes.")
                return tuple(DEFAULT_PREFIXES)

            # Extract and validate prefixes
            prefixes = set()
            for line in lines:
                line = line.strip()
                if line:
                    for prefix in line.split(","):
                        prefix = prefix.strip().upper()
                        if len(prefix) == 4 and prefix.isalpha():
                            prefixes.add(prefix)
                        else:
                            logging.debug(f"Invalid prefix skipped: {prefix}")

            _cached_prefixes = tuple(prefixes) if prefixes else tuple(DEFAULT_PREFIXES)
            return _cached_prefixes
    except FileNotFoundError:
        logging.error("File 'container_prefix.txt' not found. Using default prefixes.")
        return tuple(DEFAULT_PREFIXES)


def is_valid_carrier_prefix(container_number: str) -> bool:
    """
    Check if the container number starts with a valid carrier prefix.
    :param container_number: The recognized container number.
    :return: True if the container number starts with a valid prefix, False otherwise.
    """
    carrier_prefixes = get_carrier_prefixes()
    if not carrier_prefixes:
        logging.warning("No carrier prefixes available. Skipping prefix validation.")
        return True  # Skip validation if no prefixes are available

    prefix = container_number[:4]
    if prefix in carrier_prefixes:
        logging.debug(f"Carrier prefix '{prefix}' is valid.")
        return True
    else:
        logging.warning(f"Carrier prefix '{prefix}' is NOT valid.")
        return False


def correct_prefix(prefix: str) -> str:
    """
    Correct common misrecognitions in the carrier prefix.
    :param prefix: The first 4 characters of the container number (carrier prefix).
    :return: Corrected prefix.
    """
    # Common misrecognitions
    corrections = {
        "DDLU": "OOLU",  # Correct DDLU to OOLU
        "AOMU": "ADMU",
        # Add more corrections as needed
    }
    corrected = corrections.get(prefix, prefix)
    logging.debug(f"Corrected prefix: {corrected}")
    return corrected


def error_check(container_number: str) -> str:
    """
    Correct OCR misread errors in container numbers.
    :param container_number: The container number to fix.
    :return: The corrected container number.
    """
    if not container_number or len(container_number) != CONTAINER_LENGTH:
        logging.error("Error: Incomplete or invalid container number format!")
        return container_number

    fixed_code = list(container_number)

    # Step 1: Correct OCR misreads in the prefix (first 4 letters)
    logging.debug(f"Original Prefix: {''.join(fixed_code[:4])}")
    for i in range(4):
        fixed_code[i] = OCR_NUMBER_TO_LETTER.get(fixed_code[i], fixed_code[i])
    logging.debug(f"After OCR Corrections in Prefix: {''.join(fixed_code[:4])}")

    # Step 2: Correct OCR misreads in the serial number (next 6 digits)
    for i in range(4, 10):
        char = fixed_code[i]
        if char.isalpha():
            fixed_code[i] = OCR_LETTER_TO_NUMBER.get(char, '0')  # Default to '0' if unknown letter
        if not fixed_code[i].isdigit():
            fixed_code[i] = '0'  # Fallback to '0' if still not a digit
    logging.debug(f"After Serial Correction: {''.join(fixed_code[4:10])}")

    # Step 3: Correct checksum digit
    fixed_code[10] = OCR_LETTER_TO_NUMBER.get(fixed_code[10], fixed_code[10])
    if not fixed_code[10].isdigit():
        fixed_code[10] = '0'  # Ensure checksum is a digit

    # Step 4: Apply prefix correction
    prefix = "".join(fixed_code[:4])
    logging.debug(f"Prefix Before Correction: {prefix}")
    corrected_prefix = correct_prefix(prefix)
    logging.debug(f"Prefix After Correction: {corrected_prefix}")
    fixed_code[:4] = list(corrected_prefix)

    corrected_container = "".join(fixed_code)
    logging.debug(f"Corrected Container: {corrected_container}")
    return corrected_container


def is_valid_iso_container(container_number: str) -> bool:
    """
    Validate the given container number for conformity to ISO 6346.
    :param container_number: The container number to validate.
    :return: True if valid, False otherwise.
    """
    if not container_number:
        logging.error("Container number cannot be empty or None.")
        return False

    # Early format check: Ensure the input is 11 alphanumeric characters
    if not re.match(r'^[A-Z0-9]{11}$', container_number):
        logging.error("Invalid format: Container number must be 11 alphanumeric characters.")
        return False

    # Perform OCR corrections
    container_number = error_check(container_number)

    # Strict format check: 4 letters + 7 numbers
    if not re.match(r'^[A-Z]{4}[0-9]{7}$', container_number):
        logging.error(f"Invalid format for container number: {container_number}. Must be 4 letters followed by 7 digits.")
        return False

    # Validate carrier prefix
    if not is_valid_carrier_prefix(container_number):
        logging.error(f"Invalid carrier prefix for container number: {container_number}")
        return False

    # Validate check digit
    base_number = container_number[:10]
    try:
        given_checksum = int(container_number[10])
    except ValueError:
        logging.error(f"Invalid check digit for container number: {container_number}")
        return False

    calculated_checksum = calculate_iso_checksum(base_number)
    if given_checksum == calculated_checksum:
        logging.info(f"Valid container number: {container_number}")
        return True
    else:
        logging.error(f"Invalid check digit for container number: {container_number}. Expected {calculated_checksum}, got {given_checksum}.")
        return False


def format(container_number: str) -> str:
    """
    Reformat the container number to the standard presentation format.
    :param container_number: The container number to format.
    :return: The formatted container number.
    """
    if not is_valid_iso_container(container_number):
        raise ValueError("Invalid container number: Cannot format.")
    return ' '.join((container_number[:4], container_number[4:-1], container_number[-1:]))

'''
# Example usage
container_numbers = ["TCLU629960I", "TRHU59I5067", "INVALID123", None, ""]
for container in container_numbers:
    try:
        formatted = format(container)
        print(f"Validated and formatted: {formatted}")
    except ValueError as e:
        print(f"Error processing container number '{container}': {e}")
'''