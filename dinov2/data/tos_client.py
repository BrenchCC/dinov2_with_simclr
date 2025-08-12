import json
import os

import bytedtos
from io import BytesIO
import logging


class TosClient:
    def __init__(self, bucket_name=None, ak=None, tos_psm=None, tos_cluster=None, endpoint=None, target_delimiter="/", target_max_keys=1000, target_start_after="", timeout=10):
        self.client = bytedtos.Client(
            bucket_name, 
            ak, 
            service=tos_psm, 
            cluster=tos_cluster, 
            endpoint=endpoint,
            timeout=timeout,
            connect_timeout=timeout
        )
        self.target_delimiter = target_delimiter
        self.target_max_keys = target_max_keys
        self.target_start_after = target_start_after

    def upload_to_bucket(self, obj_key: str, content: bytes):
        if len(content) < 150:
            logging.info("skip bad img {}".format(obj_key))
            return
        content = BytesIO(content)
        try:
            resp = self.client.put_object(obj_key, content)
        except bytedtos.TosException as e:
            logging.error("upload failed. code: {}, request_id: {}, message: {}, obj_key: {}".format(e.code, e.request_id, e.msg, obj_key))

    def upload_dir_to_bucket(self, local_dir, tgt_dir, file_suffix):
        if not os.path.exists(local_dir):
            logging.error("local_dir {} not exists".format(local_dir))
            return
        for img in os.listdir(local_dir):
            if img.endswith(file_suffix):
                with open(os.path.join(local_dir, img), "rb") as f:
                    content = f.read()
                    tgt_fp = os.path.join(tgt_dir, img)
                    self.upload_to_bucket(tgt_fp, content)


    def download_file(self, item, dump_dir=None):
        item_key = item.get("key")
        item_size = item.get("size")
        item_etag = item.get("etag")
        item_last_modifed = item.get("lastModified")
        item_storage_class = item.get("storageClass")
        # logging.info(
        #     "getting item key: {}, size: {}, etag: {}, lastModified: {}, storageClass: {}".\
        #         format(item_key, item_size, item_etag, item_last_modifed, item_storage_class))
        resp = self.client.get_object(item_key)
        if len(resp.data) < 200:
            logging.info('skip bad img:{}'.format(item_key))
            return None
        sub_dir_name = os.path.dirname(item_key)
        filename = os.path.basename(item_key)
        if dump_dir:
            dump_dir = os.path.join(dump_dir, sub_dir_name)
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)
            with open(os.path.join(dump_dir, filename), "wb") as f:
                f.write(resp.data)
        return [filename, resp.data]

    def download_dir(self, tgt_dir, dump_dir=None):
        file_obj_list = []
        try:
            resp = self.client.list_prefix(
                tgt_dir, self.target_delimiter, self.target_start_after, self.target_max_keys
            )
            logging.info(resp.json)
            data = resp.json["payload"]
            objects = data["objects"]
            logging.info("action succ. code: {}, request_id: {}".format(resp.status_code, resp.headers[bytedtos.consts.ReqIdHeader]))
            if objects is None or len(objects) == 0:
                logging.error("no objects found")
                return file_obj_list

            data = resp.json["payload"]
            objects = data["objects"]
            for item in objects:
                file_obj = self.download_file(item, dump_dir)
                if file_obj:
                    file_obj_list.append(file_obj)
            return file_obj_list
        except bytedtos.TosException as e:
            logging.error("action failed. code: {}, request_id: {}, message: {}".format(e.code, e.request_id, e.msg))
            return file_obj_list
    

    def list_dir(self, tgt_dir, start_after=None):
        all_sub_dirs = []
        next_start_after, is_truncated = None, False
        start_after = start_after if start_after else self.target_start_after
        try:
            resp = self.client.list_prefix(
                    tgt_dir, self.target_delimiter, start_after, self.target_max_keys
                )
            # logging.info(resp.json)
            data = resp.json["payload"]
            next_start_after = data["startAfter"]
            all_sub_dirs = data["commonPrefix"]
            is_truncated = data["isTruncated"]
            logging.info("action succ. code: {}, request_id: {}".format(resp.status_code, resp.headers[bytedtos.consts.ReqIdHeader]))
        except:
            logger.error("List sub dirs fails")
        return all_sub_dirs, next_start_after, is_truncated

    
    def select_tos_images(self, tgt_dir, dump_dir=None):
        file_obj_list = []
        try:
            resp = self.client.list_prefix(
                tgt_dir, self.target_delimiter, self.target_start_after, self.target_max_keys
            )
            # logging.info(resp.json)
            data = resp.json["payload"]
            file_obj_list = data["objects"]
            # logging.info("action succ. code: {}, request_id: {}".format(resp.status_code, resp.headers[bytedtos.consts.ReqIdHeader]))
            # only select images within same url
            url2images = {}
            for item in file_obj_list:
                url = item["key"].split("/")[-1].split("_")[0]
                if url not in url2images:
                    url2images[url] = []
                url2images[url].append(item)
            for url, images in url2images.items():
                if len(images) > 1:
                    return [json.dumps(img, ensure_ascii=False) for img in images]

        except bytedtos.TosException as e:
            logging.error("action failed. code: {}, request_id: {}, message: {}".format(e.code, e.request_id, e.msg))
            return file_obj_list
