# pip install byted_overpass_toutiao_article_article==0.0.1
import json
import os
import requests

from overpass_toutiao_article_article.clients.rpc.toutiao_article_article import ToutiaoArticleArticleClient
from overpass_toutiao_article_article.euler_gen.cpputil.service_rpc_idl.article.article_article_thrift import GetArticleFullByGidRequest

def get_article_info_by_gid(gid):
    client = ToutiaoArticleArticleClient()
    need_fields = ["title","content_asr","content_ocr","ai_ocr_sentence","ai_asr"] # 还有asr_text, ocr_text字段也有值
    title = ""
    try:
        req_object_GetArticleFullByGid = GetArticleFullByGidRequest(GroupId=gid, AppId=13, NeedFields=need_fields)
        code, msg, resp_GetArticleFullByGid = client.GetArticleFullByGid(req_object=req_object_GetArticleFullByGid, need_resp2json=True)

        title = resp_GetArticleFullByGid['ArticleFullData']['ArticleAttrData']['Title']

        asr = resp_GetArticleFullByGid['ArticleFullData']['ArticleAttrData']['OptionalData'].get("ai_asr")
        ocr = resp_GetArticleFullByGid['ArticleFullData']['ArticleAttrData']['OptionalData'].get("ai_ocr_sentence")

        if asr is None:
            asr = resp_GetArticleFullByGid['ArticleFullData']['ArticleAttrData']['OptionalData'].get("content_asr")
            asr = clean_asr_content(asr)
        
        if ocr is None:
            ocr = resp_GetArticleFullByGid['ArticleFullData']['ArticleAttrData']['OptionalData'].get("content_ocr")
            ocr = clean_ocr_content(ocr)

        temp_dict = {
            "title": title,
            "asr": asr,
            "ocr": ocr
        }
        return temp_dict
    except Exception as e:
        print(f'{gid=}, Error:', e)
        return {"title": "", "asr": "", "ocr": ""}
    
def clean_asr_content(asr):
    asr_content = ""
    if asr is None:
        return ""
    try:
        asr = json.loads(asr)
        for item in asr:
            asr_content += asr[item]
        return asr_content
    except:
        return ""

def clean_ocr_content(ocr):
    if ocr is None:
        return ""
    
    try:
        ocr = json.loads(ocr)
        ocr_part1 = ocr.get("cover_info_ocr")[0]["Ocr"]
        ocr_part2 = ""
        ocr_temp = ocr.get("video_frame_ocr")
        if ocr_temp is not None:
            for item in ocr_temp.keys():
                for target in ocr_temp[item]:
                    ocr_part2 += target["Ocr"]
        ocr_content = ocr_part1 + ocr_part2
        return ocr_content
    except:
        return ""


def get_video_info_by_gid(gid):
    url = "https://woo9wet4.fn.bytedance.net/admin/video/gid_to_url"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "gid": int(gid)
    }
    
    response = requests.get(url, headers=headers, json=data)
    if response.status_code == 200:
        data = json.loads(response.text)
        return data
    else:
        return {}
    

def download_video_by_gid(gid, out_fp, timeout=300):
    print(f"Start downloading video with gid: {gid}")
    out_dir = os.path.dirname(out_fp)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # os.chmod(out_dir, 0o777)

    video_info = get_video_info_by_gid(gid)

    video_url = video_info['play_url']
    response = requests.get(video_url, stream=True, timeout=timeout)
    if response.status_code == 200:
        with open(out_fp, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    print(f"Finish downloading video and dump in {out_fp}")
    

if __name__ == "__main__":
    test_gid = 7510778606683947020
    timeout = 300
    resp = get_article_info_by_gid(test_gid)
    print(resp)

    download_video_by_gid(test_gid, out_fp="./temp/test.mp4")
