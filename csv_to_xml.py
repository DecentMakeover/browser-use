import csv
import os
import random
import xml.etree.ElementTree as ET
from datetime import datetime

from code_mappings import (get_code)


def create_xml():
    # Create root element with correct namespaces
    root = ET.Element('FORM15CB:FORM15CB', {
        'xmlns:Form': 'http://incometaxindiaefiling.gov.in/common',
        'xmlns:FORM15CB': 'http://incometaxindiaefiling.gov.in/FORM15CAB'
    })

    # Read CSV data
    csv_file = 'data_for_conversion.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return False

    try:
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            data = next(reader)
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return False

    # Create CreationInfo section with dynamic values
    creation_info = ET.SubElement(root, 'Form:CreationInfo')
    ET.SubElement(creation_info, 'Form:SWVersionNo').text = str(random.randint(1, 5))
    ET.SubElement(creation_info, 'Form:SWCreatedBy').text = random.choice(['DIT-EFILING-JAVA', 'DIT-EFILING-PYTHON', 'DIT-EFILING-NET'])
    ET.SubElement(creation_info, 'Form:XMLCreatedBy').text = random.choice(['DIT-EFILING-JAVA', 'DIT-EFILING-PYTHON', 'DIT-EFILING-NET'])
    ET.SubElement(creation_info, 'Form:XMLCreationDate').text = datetime.now().strftime('%Y-%m-%d')
    ET.SubElement(creation_info, 'Form:IntermediaryCity').text = random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'])

    # Create Form_Details section with dynamic values
    form_details = ET.SubElement(root, 'Form:Form_Details')
    ET.SubElement(form_details, 'Form:FormName').text = 'FORM15CB'
    ET.SubElement(form_details, 'Form:Description').text = 'FORM15CB'
    current_year = datetime.now().year
    ET.SubElement(form_details, 'Form:AssessmentYear').text = str(current_year + 1)
    ET.SubElement(form_details, 'Form:SchemaVer').text = f'Ver{random.randint(1, 3)}.{random.randint(0, 9)}'
    ET.SubElement(form_details, 'Form:FormVer').text = str(random.randint(1, 5))

    # Create RemitterDetails section
    remitter_details = ET.SubElement(root, 'FORM15CB:RemitterDetails')
    ET.SubElement(remitter_details, 'FORM15CB:IorWe').text = get_code('examiner_prefix', data.get('Examiner Prefix', ''))
    ET.SubElement(remitter_details, 'FORM15CB:RemitterHonorific').text = get_code('remitter_prefix', data.get('Remitter Prefix 1', ''))
    ET.SubElement(remitter_details, 'FORM15CB:NameRemitter').text = data.get('Remitter Name 1', '')
    ET.SubElement(remitter_details, 'FORM15CB:PAN').text = data.get('Remitter PAN TAN', '')
    ET.SubElement(remitter_details, 'FORM15CB:BeneficiaryHonorific').text = get_code('beneficiary_prefix', data.get('Beneficiary Prefix', ''))

    # Create RemitteeDetls section
    remittee_details = ET.SubElement(root, 'FORM15CB:RemitteeDetls')
    ET.SubElement(remittee_details, 'FORM15CB:NameRemittee').text = data.get('Beneficiary Name', '')

    remittee_address = ET.SubElement(remittee_details, 'FORM15CB:RemitteeAddrs')
    ET.SubElement(remittee_address, 'FORM15CB:PremisesBuildingVillage').text = data.get('Beneficiary Premises Building Village', '')
    ET.SubElement(remittee_address, 'FORM15CB:TownCityDistrict').text = data.get('Beneficiary Town City District', '')
    ET.SubElement(remittee_address, 'FORM15CB:FlatDoorBuilding').text = data.get('Beneficiary Flat Door Building', '')
    ET.SubElement(remittee_address, 'FORM15CB:AreaLocality').text = data.get('Beneficiary Area Locality', '')
    ET.SubElement(remittee_address, 'FORM15CB:ZipCode').text = data.get('Beneficiary ZIP Code', '')
    ET.SubElement(remittee_address, 'Form:State').text = data.get('Beneficiary State', '')
    ET.SubElement(remittee_address, 'FORM15CB:RoadStreet').text = data.get('Beneficiary Road Street', '')
    ET.SubElement(remittee_address, 'FORM15CB:Country').text = get_code('country', data.get('Beneficiary Country', ''))

    # Create RemittanceDetails section
    remittance_details = ET.SubElement(root, 'FORM15CB:RemittanceDetails')
    ET.SubElement(remittance_details, 'FORM15CB:CountryRemMadeSecb').text = get_code('country', data.get('Remittance Country', ''))
    ET.SubElement(remittance_details, 'FORM15CB:CurrencySecbCode').text = get_code('currency', data.get('Remittance Currency', ''))
    ET.SubElement(remittance_details, 'FORM15CB:AmtPayForgnRem').text = data.get('Amount Payable Foreign Currency', '')
    ET.SubElement(remittance_details, 'FORM15CB:AmtPayIndRem').text = data.get('Amount Payable Indian Rs', '')
    ET.SubElement(remittance_details, 'FORM15CB:NameBankCode').text = get_code('bank', data.get('Bank Name', ''))
    ET.SubElement(remittance_details, 'FORM15CB:BranchName').text = data.get('Bank Branch Name', '')
    ET.SubElement(remittance_details, 'FORM15CB:BsrCode').text = data.get('Bank BSR Code', '')
    ET.SubElement(remittance_details, 'FORM15CB:PropDateRem').text = data.get('Proposed Remittance Date', '')
    ET.SubElement(remittance_details, 'FORM15CB:NatureRemCategory').text = get_code('nature_rem_category', data.get('Nature of Remittance Agreement', ''))

    # Set RBI purpose code category and specific code
    rbi_category = data.get('RBI Purpose Code Category', '')
    rbi_specific_code = data.get('RBI Purpose Code Specific', '')
    ET.SubElement(remittance_details, 'FORM15CB:RevPurCategory').text = rbi_category
    ET.SubElement(remittance_details, 'FORM15CB:RevPurCode').text = rbi_specific_code

    # Create ItActDetails section
    it_act_details = ET.SubElement(root, 'FORM15CB:ItActDetails')
    ET.SubElement(it_act_details, 'FORM15CB:RemittanceCharIndia').text = data.get('Taxability India Flag', '')
    ET.SubElement(it_act_details, 'FORM15CB:SecRemCovered').text = data.get('Taxability India Section', '')
    ET.SubElement(it_act_details, 'FORM15CB:AmtIncChrgIt').text = data.get('Taxability India Chargeable Amount', '')
    ET.SubElement(it_act_details, 'FORM15CB:TaxLiablIt').text = data.get('Taxability India Tax Liability', '')
    ET.SubElement(it_act_details, 'FORM15CB:BasisDeterTax').text = data.get('Taxability India Basis', '')

    # Create DTAADetails section
    dtaa_details = ET.SubElement(root, 'FORM15CB:DTAADetails')
    ET.SubElement(dtaa_details, 'FORM15CB:TaxResidCert').text = data.get('DTAA TRC Obtained Flag', '')
    ET.SubElement(dtaa_details, 'FORM15CB:RelevantDtaa').text = data.get('DTAA Relevant Treaty', '')
    ET.SubElement(dtaa_details, 'FORM15CB:RelevantArtDtaa').text = data.get('DTAA Relevant Article', '')
    ET.SubElement(dtaa_details, 'FORM15CB:TaxIncDtaa').text = data.get('DTAA Taxable Income', '')
    ET.SubElement(dtaa_details, 'FORM15CB:TaxLiablDtaa').text = data.get('DTAA Tax Liability', '')
    ET.SubElement(dtaa_details, 'FORM15CB:RemForRoyFlg').text = data.get('DTAA Royalties FTS Flag', '')
    ET.SubElement(dtaa_details, 'FORM15CB:ArtDtaa').text = data.get('DTAA Royalties FTS Article', '')
    ET.SubElement(dtaa_details, 'FORM15CB:RateTdsADtaa').text = data.get('DTAA Royalties FTS TDS Rate', '')
    ET.SubElement(dtaa_details, 'FORM15CB:RemAcctBusIncFlg').text = data.get('DTAA Business Income Flag', '')
    ET.SubElement(dtaa_details, 'FORM15CB:IncLiabIndiaFlg').text = data.get('DTAA Business Income Taxable India Flag', '')
    ET.SubElement(dtaa_details, 'FORM15CB:ArrAtRateDedTax').text = data.get('DTAA Business Income Tax Rate Basis', '')
    ET.SubElement(dtaa_details, 'FORM15CB:RemOnCapGainFlg').text = data.get('DTAA Capital Gains Flag', '')
    ET.SubElement(dtaa_details, 'FORM15CB:AmtLongTrm').text = data.get('DTAA Capital Gains LTCG Amount', '')
    ET.SubElement(dtaa_details, 'FORM15CB:AmtShortTrm').text = data.get('DTAA Capital Gains STCG Amount', '')
    ET.SubElement(dtaa_details, 'FORM15CB:BasisTaxIncDtaa').text = data.get('DTAA Capital Gains Basis', '')
    ET.SubElement(dtaa_details, 'FORM15CB:OtherRemDtaa').text = data.get('DTAA Other Remittance Flag', '')
    ET.SubElement(dtaa_details, 'FORM15CB:NatureRemDtaa').text = data.get('DTAA Other Remittance Nature', '')
    ET.SubElement(dtaa_details, 'FORM15CB:TaxIndDtaaFlg').text = data.get('DTAA Other Remittance Taxable Flag', '')
    ET.SubElement(dtaa_details, 'FORM15CB:RateTdsDDtaa').text = data.get('DTAA Other Remittance TDS Rate', '')

    # Create TDSDetails section
    tds_details = ET.SubElement(root, 'FORM15CB:TDSDetails')
    ET.SubElement(tds_details, 'FORM15CB:AmtPayForgnTds').text = data.get('TDS Amount Foreign Currency', '')
    ET.SubElement(tds_details, 'FORM15CB:AmtPayIndianTds').text = data.get('TDS Amount Indian Rs', '')
    ET.SubElement(tds_details, 'FORM15CB:RateTdsSecbFlg').text = data.get('TDS Rate Basis', '')
    ET.SubElement(tds_details, 'FORM15CB:RateTdsSecB').text = data.get('TDS Rate Value', '')
    ET.SubElement(tds_details, 'FORM15CB:ActlAmtTdsForgn').text = data.get('Actual Remittance After TDS Foreign Currency', '')
    ET.SubElement(tds_details, 'FORM15CB:DednDateTds').text = data.get('TDS Deduction Date', '')

    # Create AcctntDetls section
    accountant_details = ET.SubElement(root, 'FORM15CB:AcctntDetls')
    ET.SubElement(accountant_details, 'FORM15CB:NameAcctnt').text = data.get('Accountant Name', '')
    ET.SubElement(accountant_details, 'FORM15CB:NameFirmAcctnt').text = data.get('Accountant Firm Name', '')

    accountant_address = ET.SubElement(accountant_details, 'FORM15CB:AcctntAddrs')
    ET.SubElement(accountant_address, 'FORM15CB:TownCityDistrict').text = data.get('Accountant Address Town City District', '')
    ET.SubElement(accountant_address, 'FORM15CB:FlatDoorBuilding').text = data.get('Accountant Address Line1', '')
    ET.SubElement(accountant_address, 'FORM15CB:AreaLocality').text = data.get('Accountant Address Area Locality', '')
    ET.SubElement(accountant_address, 'FORM15CB:Pincode').text = data.get('Accountant Address ZIP Code', '')
    ET.SubElement(accountant_address, 'Form:State').text = data.get('Accountant Address Type', '')
    ET.SubElement(accountant_address, 'FORM15CB:Country').text = get_code('country', data.get('Accountant Address Country', ''))

    ET.SubElement(accountant_details, 'FORM15CB:MembershipNumber').text = data.get('Accountant Membership No', '')
    ET.SubElement(accountant_details, 'FORM15CB:RegNoAcctnt').text = data.get('Accountant Registration No', '')

    # Write XML to file with proper formatting
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")

    # Write to file with XML declaration
    with open('output.xml', 'wb') as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n')
        tree.write(f, encoding='utf-8')


def main():
    """Convert CSV to XML."""
    try:
        create_xml()
        print("XML file created successfully at output.xml")
    except Exception as e:
        print(f"Error creating XML: {str(e)}")


if __name__ == "__main__":
    main()
