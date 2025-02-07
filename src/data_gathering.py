import json
from pathlib import Path
import requests

# declaring path
raw_data_path = Path(__file__).parent.parent / "data" / "raw"
processed_data_path = Path(__file__).parent.parent / "data" / "processed"

# declaring url
api_url = "https://ckandev.indiadataportal.com/api/3/action/package_search?q=organization%3Aidp-organization&rows=1000"
headers = {
    'Content-Type': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
# Make the request
response = requests.get(api_url, headers=headers)

if response.status_code != 200:
    print(f"Error: Received HTTP {response.status_code}")
    print(f"Response text: {response.text}")
else:
    try:
        print("success")
        json_data = response.json()
    except requests.exceptions.JSONDecodeError:
        print("Error: Response is not valid JSON")
        print(f"Raw response: {response.text}")

if response.status_code == 200 and json_data.get('success'):
    packages = json_data['result']['results']

    for package in packages:
        # create a directory for each package in raw_data_path
        package_dir = raw_data_path / package['name']
        package_dir.mkdir(parents=True, exist_ok=True)

        # write the package data to a file in the package directory
        package_file = package_dir / f'{package["name"]}.json'
        with package_file.open('w') as f:
            json.dump(package, f, indent=4)
        for resource in package['resources']:
            # create a directory for each resource in the package directory
            resource_dir = package_dir / resource['sku']
            resource_dir.mkdir(parents=True, exist_ok=True)

            # make another api call for datastore_info
            datastore_info_url = f"https://ckandev.indiadataportal.com/api/3/action/datastore_info?id={resource['id']}"
            datastore_info_response = requests.get(datastore_info_url, headers=headers)
            # if response is not 200, print the error
            if datastore_info_response.status_code != 200:
                print(f"Error: Received HTTP {datastore_info_response.status_code}")
                print(f"Response text: {datastore_info_response.text}")
            else:
                try:
                    datastore_info = datastore_info_response.json()
                except requests.exceptions.JSONDecodeError:
                    print("Error: Response is not valid JSON")
                    print(f"Raw response: {datastore_info_response.text}")

            if datastore_info_response.status_code == 200 and datastore_info.get('success'):
                # write the datastore_info data to a file in the resource directory
                datastore_info_file = resource_dir / 'datastore_info.json'
                with datastore_info_file.open('w') as f:
                    json.dump(datastore_info, f, indent=4)


                