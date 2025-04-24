import csv
from validation import validate_csv_data

# Read the form_fields_with_data.csv
with open('form_fields_with_data.csv', 'r', newline='') as input_file:
    csv_reader = csv.reader(input_file)
    header = next(csv_reader)  # Skip the header row
    
    # Read the field names and values
    field_names = []
    values = []
    for row in csv_reader:
        if len(row) >= 2:
            field_names.append(row[0])
            values.append(row[1])

# Create the data_for_conversion.csv
with open('data_for_conversion.csv', 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(field_names)  # Write field names as the header row
    csv_writer.writerow(values)      # Write values as the data row

print('Created data_for_conversion.csv successfully')

# Now validate it
print("Validation results:")
print(validate_csv_data('data_for_conversion.csv')) 