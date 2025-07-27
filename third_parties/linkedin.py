import requests


def scrape_linkedin_profile(profile_url: str):
    response = requests.get(profile_url, timeout=20)
    data = response.json()
    return data


if __name__ == "__main__":
    print(
        scrape_linkedin_profile(
            "https://gist.githubusercontent.com/hpdeveloper28/5b66ccb34c2dc02ebea197bfe3d3680e/raw/52020e549c901224d372396b35318957fbb3bd36/sample_linkedin.json"
        )
    )
