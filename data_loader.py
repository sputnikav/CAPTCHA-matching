import json
import os
import urllib.request

import cv2 as cv
import numpy as np
import requests
from PIL import Image

from settings import CAPTCHA_ID, CHALLENGE_ID


def geetest_resources_extractor(captcha_id, challenge_id):
    """Extract Resources from Geetest

    Args:
        captcha_id (_type_): CAPTCHA ID
        challenge_id (_type_): Challenge ID

    Returns:
        _type_: _description_
    """
    r = requests.get(
        "https://gcaptcha4.geetest.com/load",
        params={
            "captcha_id": captcha_id,
            "challenge": challenge_id,
            "client_type": "web",
            "risk_type": "icon",
            "lang": "en",
        },
    )
    r.raise_for_status()
    assert r.text.startswith("(")
    assert r.text.endswith(")")
    data = json.loads(r.text.lstrip("(").rstrip(")"))
    return {
        "lot_number": data["data"]["lot_number"],
        "bg": "https://static.geetest.com/%s" % data["data"]["imgs"],
        "i1": "https://static.geetest.com/%s" % data["data"]["ques"][0],
        "i2": "https://static.geetest.com/%s" % data["data"]["ques"][1],
        "i3": "https://static.geetest.com/%s" % data["data"]["ques"][2],
    }


def url_to_image(url):
    """Convert URL to image

    Args:
        url (str): URL

    Returns:
        array: image array decoded by OpenCV
    """
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_UNCHANGED)
    return image


def denoise_image(image):
    """Remove noise from image

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    se = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
    bg = cv.morphologyEx(image, cv.MORPH_DILATE, se)
    out_gray = cv.divide(image, bg, scale=255)
    out_binary = cv.threshold(out_gray, 0, 255, cv.THRESH_OTSU)[1]
    cv.imshow("binary", out_binary)
    cv.imwrite("binary.png", out_binary)
    cv.imshow("gray", out_gray)
    cv.imwrite("gray.png", out_gray)
    return out_gray


def remove_dots(image):
    """Remove dots from image

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    src = cv.imread(image, cv.IMREAD_GRAYSCALE)
    # convert to binary by thresholding
    ret, binary_map = cv.threshold(src, 127, 255, 0)

    # do connected components processing
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_map, None, None, None, 8, cv.CV_32S)

    # get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:, cv.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 100:  # keep
            result[labels == i + 1] = 255
    return result


def threshold_image(image):
    """Threshold to generate binary image

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_channels = image.shape[2]

    if n_channels > 3:
        single_channel = image[:, :, 3]
    else:
        single_channel = np.maximum.reduce([image[:, :, 0], image[:, :, 1], image[:, :, 2]])
    ret, img = cv.threshold(single_channel, 254, 255, cv.THRESH_BINARY)

    return img


def display_images_sift(bg, i1, i2, i3, bg_kp, i1_kp, i2_kp, i3_kp):
    """Display Image for processing using SIFT

    Args:
        bg (_type_): _description_
        i1 (_type_): _description_
        i2 (_type_): _description_
        i3 (_type_): _description_
        bg_kp (_type_): _description_
        i1_kp (_type_): _description_
        i2_kp (_type_): _description_
        i3_kp (_type_): _description_
    """
    bg_wkp = cv.drawKeypoints(bg, bg_kp, bg, (0, 255, 0), 4)
    i1_wkp = cv.drawKeypoints(i1, i1_kp, i1, (0, 255, 0), 4)
    i2_wkp = cv.drawKeypoints(i2, i2_kp, i2, (0, 255, 0), 4)
    i3_wkp = cv.drawKeypoints(i3, i3_kp, i3, (0, 255, 0), 4)
    icons = np.hstack(
        [
            cv.copyMakeBorder(i1_wkp, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=255),
            cv.copyMakeBorder(i2_wkp, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=255),
            cv.copyMakeBorder(i3_wkp, 1, 1, 1, 1 + 146, cv.BORDER_CONSTANT, value=255),
        ]
    )

    img = np.vstack([bg_wkp, icons])
    cv.imshow("Geetest", img)


def display_images_orb(bg, i1, i2, i3, bg_kp, i1_kp, i2_kp, i3_kp):
    """Display Image for processing using ORB

    Args:
        bg (_type_): _description_
        i1 (_type_): _description_
        i2 (_type_): _description_
        i3 (_type_): _description_
        bg_kp (_type_): _description_
        i1_kp (_type_): _description_
        i2_kp (_type_): _description_
        i3_kp (_type_): _description_
    """
    bg_wkp = cv.drawKeypoints(bg, bg_kp, bg, (0, 255, 0), 4)
    i1_wkp = cv.drawKeypoints(i1, i1_kp, i1, (0, 255, 0), 4)
    i2_wkp = cv.drawKeypoints(i2, i2_kp, i2, (0, 255, 0), 4)
    i3_wkp = cv.drawKeypoints(i3, i3_kp, i3, (0, 255, 0), 4)
    icons = np.vstack(
        [
            cv.copyMakeBorder(i1_wkp, 0, 0, 0, 252, cv.BORDER_CONSTANT, value=255),
            cv.copyMakeBorder(i2_wkp, 0, 0, 0, 252, cv.BORDER_CONSTANT, value=255),
            cv.copyMakeBorder(i3_wkp, 0, 0, 0, 252, cv.BORDER_CONSTANT, value=255),
        ]
    )

    img = np.vstack([bg_wkp, icons])
    cv.imshow("Geetest", img)


def repeat_downloading_and_saving_images(captcha_id, challenge_id, count, method="orb"):
    """Download and save background + icons images with counting

    Args:
        captcha_id (_type_): _description_
        challenge_id (_type_): _description_
        count (_type_): _description_
        method (str, optional): _description_. Defaults to "orb".
    """
    data = geetest_resources_extractor(captcha_id, challenge_id)
    urls = [data["bg"], data["i1"], data["i2"], data["i3"]]
    raw_imgs = list(map(url_to_image, urls))

    if method == "orb":
        output_folder = f"./images_orb_new_{count}"
        imgs = list(
            map(
                lambda img: cv.copyMakeBorder(threshold_image(img), 200, 200, 200, 200, cv.BORDER_CONSTANT, value=0),
                raw_imgs,
            )
        )
    else:
        output_folder = f"./images_sift_new_{count}"
        imgs = list(
            map(lambda img: cv.copyMakeBorder(threshold_image(img), 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0), raw_imgs)
        )
    os.makedirs(output_folder, exist_ok=True)

    def save_arr_to_img(arr, path):
        cv.imwrite(path, arr)

    ## save the original background
    path_raw = os.path.join(output_folder, "raw_background.png")

    save_arr_to_img(
        arr=imgs[0],
        path=path_raw,
    )

    ## save the denoised background
    path_denoised = os.path.join(output_folder, "background.png")
    img_denoised = remove_dots(path_raw)
    save_arr_to_img(arr=img_denoised, path=path_denoised)

    for i in [1, 2, 3]:
        save_arr_to_img(arr=imgs[i], path=os.path.join(output_folder, f"item_{i}.png"))
    return


def load_data_from_images_folders(path, name_tag="background"):
    """Load data from images folders

    Args:
        path (_type_): _description_
        name_tag (str, optional): _description_. Defaults to "background".

    Returns:
        _type_: _description_
    """
    background_img = np.asarray(Image.open(os.path.join(path, f"{name_tag}.png")))
    # background_img.show()
    ret_list = [background_img]
    ret_list += [np.asarray(Image.open(os.path.join(path, f"item_{i}.png"))) for i in [1, 2, 3]]
    return ret_list


if __name__ == "__main__":
    print("start..")
    ## download and store images for evaluation
    for i in range(100):
        print(f"downloading {i}-th sample")
        repeat_downloading_and_saving_images(
            captcha_id=os.getenv(CAPTCHA_ID), challenge_id=os.getenv(CHALLENGE_ID), count=i, method="sift"
        )
    print("done..")
