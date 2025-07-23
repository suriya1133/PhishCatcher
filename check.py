import re
from urllib.parse import urlparse

def extract_features_from_url(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    return {
        "NumDots": url.count('.'),
        "SubdomainLevel": hostname.count('.') - 1,
        "PathLevel": path.count('/'),
        "UrlLength": len(url),
        "NumDash": url.count('-'),
        "NumDashInHostname": hostname.count('-'),
        "AtSymbol": int('@' in url),
        "TildeSymbol": int('~' in url),
        "NumUnderscore": url.count('_'),
        "NumPercent": url.count('%'),
        "NumQueryComponents": query.count('&') + 1 if query else 0,
        "NumAmpersand": url.count('&'),
        "NumHash": url.count('#'),
        "NumNumericChars": len(re.findall(r'\d', url)),
        "NoHttps": int(not url.startswith("https")),
        "RandomString": int(len(re.findall(r'[a-zA-Z]{10,}', hostname)) > 0),
        "IpAddress": int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname))),
        "DomainInSubdomains": 0,
        "DomainInPaths": 0,
        "HttpsInHostname": int('https' in hostname),
        "HostnameLength": len(hostname),
        "PathLength": len(path),
        "QueryLength": len(query),
        "DoubleSlashInPath": int('//' in path),
        "NumSensitiveWords": sum(word in url.lower() for word in ["secure", "account", "update", "login", "verify"]),
        "EmbeddedBrandName": 0,
        "PctExtHyperlinks": 0,
        "PctExtResourceUrls": 0,
        "ExtFavicon": 0,
        "InsecureForms": 0,
        "RelativeFormAction": 0,
        "ExtFormAction": 0,
        "AbnormalFormAction": 0,
        "PctNullSelfRedirectHyperlinks": 0,
        "FrequentDomainNameMismatch": 0,
        "FakeLinkInStatusBar": 0,
        "RightClickDisabled": 0,
        "PopUpWindow": 0,
        "SubmitInfoToEmail": 0,
        "IframeOrFrame": 0,
        "MissingTitle": 0,
        "ImagesOnlyInForm": 0,
        "SubdomainLevelRT": 0,
        "UrlLengthRT": 0,
        "PctExtResourceUrlsRT": 0,
        "AbnormalExtFormActionR": 0,
        "ExtMetaScriptLinkRT": 0,
        "PctExtNullSelfRedirectHyperlinksRT": 0
    }
