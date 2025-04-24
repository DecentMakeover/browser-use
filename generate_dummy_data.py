import csv
import random
import datetime
import os
import time
from faker import Faker
from code_mappings import (
    get_code, COUNTRY_MAP, CURRENCY_MAP, BANK_MAP, NATURE_REM_CATEGORY_MAP,
    RBI_PURPOSE_CATEGORY_MAP, RBI_PURPOSE_CODE_MAP, get_rbi_purpose_code
)

def get_unique_seed():
    """Generate a unique seed using microsecond precision."""
    now = datetime.datetime.now()
    return int(now.strftime('%Y%m%d%H%M%S%f'))

def get_fake_data():
    """Get a new Faker instance with a unique seed."""
    seed = get_unique_seed()
    random.seed(seed)
    fake = Faker()
    fake.seed_instance(seed)
    return fake

def generate_dummy_data():
    # Create a dictionary to hold the data
    data = {}
    
    # Form header data
    examiner_prefix = random.choice(['I', 'We'])
    data['Examiner Prefix'] = examiner_prefix  # Store human-readable value
    
    remitter_prefix = random.choice(['Mr', 'Ms', 'M/s', 'Dr'])
    data['Remitter Prefix 1'] = remitter_prefix  # Store human-readable value
    data['Remitter Name 1'] = get_fake_data().company() if remitter_prefix == 'M/s' else get_fake_data().name()
    data['Remitter PAN TAN'] = get_fake_data().lexify(text='?????####?')
    
    beneficiary_prefix = random.choice(['Mr', 'Ms', 'M/s'])
    data['Beneficiary Prefix'] = beneficiary_prefix  # Store human-readable value
    data['Beneficiary Name Header'] = get_fake_data().name()
    
    # Beneficiary details
    data['Beneficiary Name'] = get_fake_data().name()
    data['Beneficiary Flat Door Building'] = str(random.randint(1, 999))
    data['Beneficiary Premises Building Village'] = get_fake_data().company()
    data['Beneficiary Road Street'] = get_fake_data().street_address()
    data['Beneficiary Area Locality'] = get_fake_data().word()
    data['Beneficiary Town City District'] = get_fake_data().city()
    data['Beneficiary State'] = get_fake_data().state()
    
    # Use a random country from our mapping
    country = random.choice(list(COUNTRY_MAP.keys()))
    data['Beneficiary Country'] = country  # Store human-readable value
    data['Beneficiary ZIP Code'] = get_fake_data().postcode()
    
    # Remittance details
    remittance_country = random.choice(list(COUNTRY_MAP.keys()))
    data['Remittance Country'] = remittance_country  # Store human-readable value
    
    currency = random.choice(list(CURRENCY_MAP.keys()))
    data['Remittance Currency'] = currency  # Store human-readable value
    
    data['Amount Payable Foreign Currency'] = str(round(random.uniform(1000, 100000), 2))
    data['Amount Payable Indian Rs'] = str(round(float(data['Amount Payable Foreign Currency']) * random.uniform(70, 85), 2))
    
    bank = random.choice(list(BANK_MAP.keys()))
    data['Bank Name'] = bank  # Store human-readable value
    data['Bank Branch Name'] = get_fake_data().city()[:10]
    data['Bank BSR Code'] = str(random.randint(1000000, 9999999))
    
    # Generate a future date for remittance
    future_date = datetime.datetime.now() + datetime.timedelta(days=random.randint(10, 60))
    data['Proposed Remittance Date'] = future_date.strftime('%Y-%m-%d')
    
    # Use a random nature remittance category from our mapping
    nature_rem_category = random.choice(list(NATURE_REM_CATEGORY_MAP.keys()))
    data['Nature of Remittance Agreement'] = nature_rem_category  # Store human-readable value
    
    # Generate RBI purpose code category and specific code
    category_name = random.choice(list(RBI_PURPOSE_CATEGORY_MAP.keys()))
    category_code = category_name  # Use the category code directly (like 'RB-5.1')

    # Get list of specific codes for this category
    specific_codes = list(RBI_PURPOSE_CODE_MAP[category_code].keys())
    specific_code = random.choice(specific_codes)

    data['RBI Purpose Code Category'] = category_code
    data['RBI Purpose Code Specific'] = specific_code
    data['Grossed Up Flag'] = random.choice(['Y', 'N'])
    
    # Taxability details
    data['Taxability India Flag'] = random.choice(['Y', 'N'])
    if data['Taxability India Flag'] == 'N':
        data['Taxability India Reason Not Taxable'] = "Not taxable as per section " + str(random.randint(1, 30))
        data['Taxability India Section'] = ''
        data['Taxability India Chargeable Amount'] = ''
        data['Taxability India Tax Liability'] = ''
        data['Taxability India Basis'] = ''
    else:
        data['Taxability India Reason Not Taxable'] = ''
        data['Taxability India Section'] = str(random.randint(1, 200))
        data['Taxability India Chargeable Amount'] = str(round(random.uniform(1000, 50000), 2))
        data['Taxability India Tax Liability'] = str(round(float(data['Taxability India Chargeable Amount']) * 0.1, 2))
        data['Taxability India Basis'] = get_fake_data().text(max_nb_chars=20)
    
    # DTAA details
    data['DTAA Relief Claimed Flag'] = random.choice(['Y', 'N'])
    if data['DTAA Relief Claimed Flag'] == 'Y':
        data['DTAA TRC Obtained Flag'] = random.choice(['Y', 'N'])
        data['DTAA Relevant Treaty'] = str(random.randint(100, 500))
        data['DTAA Relevant Article'] = str(random.randint(1, 30))
        data['DTAA Taxable Income'] = str(round(random.uniform(5000, 80000), 2))
        data['DTAA Tax Liability'] = str(round(float(data['DTAA Taxable Income']) * 0.08, 2))
        
        # Royalties/FTS details
        data['DTAA Royalties FTS Flag'] = random.choice(['Y', 'N'])
        if data['DTAA Royalties FTS Flag'] == 'Y':
            data['DTAA Royalties FTS Article'] = str(random.randint(1, 20))
            data['DTAA Royalties FTS TDS Rate'] = str(round(random.uniform(0.05, 0.2), 2))
        else:
            data['DTAA Royalties FTS Article'] = ''
            data['DTAA Royalties FTS TDS Rate'] = ''
        
        # Business income details
        data['DTAA Business Income Flag'] = random.choice(['Y', 'N'])
        if data['DTAA Business Income Flag'] == 'Y':
            data['DTAA Business Income Taxable India Flag'] = random.choice(['Y', 'N'])
            if data['DTAA Business Income Taxable India Flag'] == 'Y':
                data['DTAA Business Income Tax Rate Basis'] = random.choice(['YES', 'NO', 'Other'])
                data['DTAA Business Income Not Taxable Reason'] = ''
            else:
                data['DTAA Business Income Tax Rate Basis'] = ''
                data['DTAA Business Income Not Taxable Reason'] = get_fake_data().text(max_nb_chars=50)
        else:
            data['DTAA Business Income Taxable India Flag'] = ''
            data['DTAA Business Income Tax Rate Basis'] = ''
            data['DTAA Business Income Not Taxable Reason'] = ''
        
        # Capital gains details
        data['DTAA Capital Gains Flag'] = random.choice(['Y', 'N'])
        if data['DTAA Capital Gains Flag'] == 'Y':
            data['DTAA Capital Gains LTCG Amount'] = str(round(random.uniform(10000, 200000), 2))
            data['DTAA Capital Gains STCG Amount'] = str(round(random.uniform(5000, 100000), 2))
            data['DTAA Capital Gains Basis'] = get_fake_data().text(max_nb_chars=40)
        else:
            data['DTAA Capital Gains LTCG Amount'] = ''
            data['DTAA Capital Gains STCG Amount'] = ''
            data['DTAA Capital Gains Basis'] = ''
        
        # Other remittance details
        data['DTAA Other Remittance Flag'] = random.choice(['Y', 'N'])
        if data['DTAA Other Remittance Flag'] == 'Y':
            data['DTAA Other Remittance Nature'] = get_fake_data().text(max_nb_chars=30)
            data['DTAA Other Remittance Taxable Flag'] = random.choice(['Y', 'N'])
            if data['DTAA Other Remittance Taxable Flag'] == 'Y':
                data['DTAA Other Remittance TDS Rate'] = str(round(random.uniform(0.1, 0.3), 2))
                data['DTAA Other Remittance Not Taxable Reason'] = ''
            else:
                data['DTAA Other Remittance TDS Rate'] = ''
                data['DTAA Other Remittance Not Taxable Reason'] = get_fake_data().text(max_nb_chars=50)
        else:
            data['DTAA Other Remittance Nature'] = ''
            data['DTAA Other Remittance Taxable Flag'] = ''
            data['DTAA Other Remittance TDS Rate'] = ''
            data['DTAA Other Remittance Not Taxable Reason'] = ''
    else:
        # If DTAA relief not claimed, set all DTAA fields to empty
        dtaa_fields = [
            'DTAA TRC Obtained Flag', 'DTAA Relevant Treaty', 'DTAA Relevant Article',
            'DTAA Taxable Income', 'DTAA Tax Liability', 'DTAA Royalties FTS Flag',
            'DTAA Royalties FTS Article', 'DTAA Royalties FTS TDS Rate',
            'DTAA Business Income Flag', 'DTAA Business Income Taxable India Flag',
            'DTAA Business Income Tax Rate Basis', 'DTAA Business Income Not Taxable Reason',
            'DTAA Capital Gains Flag', 'DTAA Capital Gains LTCG Amount',
            'DTAA Capital Gains STCG Amount', 'DTAA Capital Gains Basis',
            'DTAA Other Remittance Flag', 'DTAA Other Remittance Nature',
            'DTAA Other Remittance Taxable Flag', 'DTAA Other Remittance TDS Rate',
            'DTAA Other Remittance Not Taxable Reason'
        ]
        for field in dtaa_fields:
            data[field] = ''
    
    # TDS details
    tds_amount_foreign = str(round(float(data['Amount Payable Foreign Currency']) * 0.1, 2)) if data.get('Taxability India Flag') == 'Y' else '0'
    data['TDS Amount Foreign Currency'] = tds_amount_foreign
    data['TDS Amount Indian Rs'] = str(round(float(tds_amount_foreign) * float(data['Amount Payable Indian Rs']) / float(data['Amount Payable Foreign Currency']), 2))
    data['TDS Rate Basis'] = random.choice(['1', '2'])  # 1=AS PER INCOME TAX ACT, 2=AS PER DTAA
    data['TDS Rate Value'] = str(random.choice([0, 0.05, 0.1, 0.15, 0.2, 0.3]))
    
    # Calculate actual remittance after TDS
    foreign_amt = float(data['Amount Payable Foreign Currency'])
    tds_amt = float(tds_amount_foreign)
    data['Actual Remittance After TDS Foreign Currency'] = str(round(foreign_amt - tds_amt, 2))
    
    # TDS deduction date
    tds_date = datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 10))
    data['TDS Deduction Date'] = tds_date.strftime('%Y-%m-%d')
    
    # Accountant details
    data['Accountant Name'] = get_fake_data().name()
    data['Accountant Firm Name'] = get_fake_data().company()
    data['Accountant Address Line1'] = str(random.randint(1, 999))
    data['Accountant Address Premises Building Village'] = get_fake_data().company()
    data['Accountant Address Road Street'] = get_fake_data().street_address()
    data['Accountant Address Area Locality'] = get_fake_data().word()
    data['Accountant Address Town City District'] = get_fake_data().city()
    data['Accountant Address Type'] = random.choice(['Domestic', 'Foreign'])
    
    accountant_country = 'India' if data['Accountant Address Type'] == 'Domestic' else random.choice(list(COUNTRY_MAP.keys()))
    data['Accountant Address Country'] = accountant_country  # Store human-readable value
    data['Accountant Address ZIP Code'] = get_fake_data().postcode()
    data['Accountant Membership No'] = str(random.randint(100000, 999999))
    data['Accountant Registration No'] = str(random.randint(10000000, 99999999))
    
    return data

def generate_csv_with_dummy_data(output_csv_path='form_fields_with_data.csv'):
    # Clear any existing intermediate files
    if os.path.exists('data_for_conversion.csv'):
        os.remove('data_for_conversion.csv')
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)
    
    # Generate dummy data
    dummy_data = generate_dummy_data()
    
    # Create CSV with dummy data
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write header row
        writer.writerow(['Field Name', 'Value'])
        
        # Write data rows
        for field, value in dummy_data.items():
            writer.writerow([field, value])
    
    print(f"CSV with dummy data created at {output_csv_path}")
    
    # Create a properly formatted CSV that our converter can use
    with open('data_for_conversion.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=dummy_data.keys())
        writer.writeheader()
        writer.writerow(dummy_data)
    
    print("Conversion-ready CSV created at data_for_conversion.csv")

def main():
    """Generate dummy data and save to CSV."""
    try:
        # Generate the data
        print("Generating dummy data...")
        data = generate_dummy_data()
        print("✓ Data generated")
        
        # Write to form_fields_with_data.csv
        print("Writing to form_fields_with_data.csv...")
        with open('form_fields_with_data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Field Name', 'Value'])
            for field, value in data.items():
                writer.writerow([field, value])
        print("✓ Generated form_fields_with_data.csv")
        
        # Write to data_for_conversion.csv
        print("Writing to data_for_conversion.csv...")
        with open('data_for_conversion.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)
        print("✓ Generated data_for_conversion.csv")
        
        return True
    except Exception as e:
        import traceback
        print(f"Error generating dummy data: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Check if the requirements are installed
    try:
        import faker
    except ImportError:
        print("The 'faker' package is required but not installed.")
        print("Please install it using: pip install faker")
        exit(1)
    
    main() 