import re
import urllib
import socket
import requests
import tldextract
import whois
from datetime import datetime

def has_ip(url):
    return 1 if re.match(r"(http[s]?://)?(\d{1,3}\.){3}\d{1,3}", url) else 0

def abnormal_url(url):
    hostname = urllib.parse.urlparse(url).hostname
    return 1 if hostname not in url else 0

def google_index(url):
    try:
        response = requests.get(f"https://www.google.com/search?q=site:{url}", timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        return 0 if "did not match any documents" in response.text else 1
    except:
        return 0

def web_traffic(url):
    try:
        domain = urllib.parse.urlparse(url).netloc
        response = requests.get(f"https://data.alexa.com/data?cli=10&dat=s&url={domain}", timeout=5)
        if "<REACH" in response.text:
            return 1
        else:
            return 0
    except:
        return 0

def domain_age(domain):
    try:
        info = whois.whois(domain)
        creation_date = info.creation_date
        expiration_date = info.expiration_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        age = (expiration_date - creation_date).days
        return 0 if age <= 365 else 1
    except:
        return 0

def extract_features_from_url(url):
    parsed = urllib.parse.urlparse(url)
    ext = tldextract.extract(url)
    domain = ext.domain + "." + ext.suffix
    subdomain = ext.subdomain

    features = {
        "Having_IP_Address": has_ip(url),
        "URL_Length": len(url),
        "Shortining_Service": 1 if re.search(r"(bit\.ly|goo\.gl|tinyurl\.com|ow\.ly)", url) else 0,
        "Having_At_Symbol": 1 if "@" in url else 0,
        "Prefix_Suffix": 1 if "-" in domain else 0,
        "Subdomain_Level": len(subdomain.split(".")) if subdomain else 0,
        "Abnormal_URL": abnormal_url(url),
        "DNSRecord": 0 if domain_age(domain) == 0 else 1,
        "Web_Traffic": web_traffic(url),
        "Google_Index": google_index(url),
        "Domain_registeration_length": domain_age(domain),
    }

    return features
