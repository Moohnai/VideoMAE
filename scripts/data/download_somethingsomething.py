import os

file_adresses = [
    "https://developer.qualcomm.com/qfile/68975/20bn-something-something-v2-00.zip",
    "https://developer.qualcomm.com/qfile/68976/20bn-something-something-v2-01.zip",
    "https://developer.qualcomm.com/qfile/68977/20bn-something-something-v2-02.zip",
    "https://developer.qualcomm.com/qfile/68978/20bn-something-something-v2-03.zip",
    "https://developer.qualcomm.com/qfile/68979/20bn-something-something-v2-04.zip",
    "https://developer.qualcomm.com/qfile/68980/20bn-something-something-v2-05.zip",
    "https://developer.qualcomm.com/qfile/68981/20bn-something-something-v2-06.zip",
    "https://developer.qualcomm.com/qfile/68982/20bn-something-something-v2-07.zip",
    "https://developer.qualcomm.com/qfile/68983/20bn-something-something-v2-08.zip",
    "https://developer.qualcomm.com/qfile/68984/20bn-something-something-v2-09.zip",
    "https://developer.qualcomm.com/qfile/68985/20bn-something-something-v2-10.zip",
    "https://developer.qualcomm.com/qfile/68986/20bn-something-something-v2-11.zip",
    "https://developer.qualcomm.com/qfile/68987/20bn-something-something-v2-12.zip",
    "https://developer.qualcomm.com/qfile/68988/20bn-something-something-v2-13.zip",
    "https://developer.qualcomm.com/qfile/68989/20bn-something-something-v2-14.zip",
    "https://developer.qualcomm.com/qfile/68990/20bn-something-something-v2-15.zip",
    "https://developer.qualcomm.com/qfile/68991/20bn-something-something-v2-16.zip",
    "https://developer.qualcomm.com/qfile/68992/20bn-something-something-v2-17.zip",
    "https://developer.qualcomm.com/qfile/68993/20bn-something-something-v2-18.zip",
    "https://developer.qualcomm.com/qfile/68994/20bn-something-something-v2-19.zip",
    "https://developer.qualcomm.com/qfile/68943/20bn-something-something-download-package-labels.zip"
]



if not os.path.exists('/home/mona/somethingsomething/'):
    os.mkdir('/home/mona/somethingsomething/')

for adress in file_adresses:
    file_name = adress.split('/')[-1]
    if os.path.exists(f'/home/mona/somethingsomething/{file_name}'):
        continue
    else:
        download_script = f"""curl {adress} \
        -H 'authority: developer.qualcomm.com' \
        -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
        -H 'accept-language: en-GB,en-US;q=0.9,en;q=0.8' \
        -H 'cache-control: max-age=0' \
        -H 'cookie: INGRESSCOOKIE=1664794343.813.1701.796836|b1b41e039a516dae4a16dde1372167d1; s_cc=true; OptanonAlertBoxClosed=2022-10-03T10:52:28.802Z; OptanonConsent=isGpcEnabled=0&datestamp=Mon+Oct+03+2022+11%3A52%3A28+GMT%2B0100+(British+Summer+Time)&version=6.39.0&isIABGlobal=false&hosts=&consentId=8555a33b-f042-439b-96da-db367b09ebe3&interactionCount=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0003%3A1%2CC0004%3A1; AMCV_14DFEF2E54411B460A4C98A6%40AdobeOrg=1999109931%7CMCMID%7C89617323272237949262238593270338942478%7CMCAAMLH-1665399154%7C6%7CMCAAMB-1665399154%7C6G1ynYcLPuiQxYZrsz_pkqfLG9yMXBpb2zX5dvJdYQJzPXImdj0y%7CMCAID%7CNONE; utag_main=_st:1664796157021$ses_id:1664795106333%3Bexp-session$v_id:01839d79e4110053e6954036ac280506f02a706700978$_sn:1$_se:2$_ss:0$_pn:1%3Bexp-session$vapi_domain:qualcomm.com; s_sq=qualcomm.us.qdn.prod%3D%2526pid%253Dqdn%25253Aqfile%25253A68975%25253A20bn-something-something-v2-00.zip%2526pidt%253D1%2526oid%253Dhttps%25253A%25252F%25252Fdeveloper.qualcomm.com%25252Fuser%25253Fdestination%25253Dqfile%25252F68975%25252F20bn-something-something-v2-00.zip%2526ot%253DA; SESSe9b825fe435fd0ce0540e1ea73912b52=9zFlQ618qoCsOQoV_RHQaM7LvflRPeJTacO0p4B1Mag; QUSERNAME=m.ahmadian' \
        -H 'referer: https://stepup.qualcomm.com/' \
        -H 'sec-ch-ua: "Chromium";v="106", "Google Chrome";v="106", "Not;A=Brand";v="99"' \
        -H 'sec-ch-ua-mobile: ?0' \
        -H 'sec-ch-ua-platform: "Windows"' \
        -H 'sec-fetch-dest: document' \
        -H 'sec-fetch-mode: navigate' \
        -H 'sec-fetch-site: same-site' \
        -H 'upgrade-insecure-requests: 1' \
        -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36' \
        --compressed \
        --output ./somethingsomething/{file_name}"""

        os.system(download_script)

# os.system('unzip /home/mona/somethingsomething/20bn-something-something-v2-\??.zip')
# os.system('cat 20bn-something-something-v2-?? | tar -xvzf -')

dir_path = r'20bn-something-something-v2'
print("Total number of videos: ", len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))