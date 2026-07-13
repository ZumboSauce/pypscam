import os
import time

import numpy as np
import usb

VENDOR_ID = 0x05A9
BOOT_PRODUCT_ID = 0x0580  # boot mode (important)
RUN_PRODUCT_ID = 0x0D0A  # already initialized (skip)

FRAME_SIZE = 3448 * 2 * 808

FIRMWARE_PATH = "firmware.bin"


def find_usb(idVendor, idProduct):
    if dev := usb.core.find(idVendor=idVendor, idProduct=idProduct):
        if isinstance(dev, usb.core.Device):
            return dev
    return None


def upload(pc: usb.core.Device, path: str):
    pc.set_configuration()
    with open(path, "rb") as fw:
        print("opened")
        wValue = 0
        wIndex = 0x14
        while chunk := fw.read(512):
            pc.ctrl_transfer(0x40, 0x00, wValue, wIndex, chunk)
            if (wValue := wValue + 512) >= 0x10000:
                wValue = 0
                wIndex += 1
    try:
        pc.ctrl_transfer(0x40, 0x00, 0x2200, 0x8018, [0x5B])
    except usb.core.USBError:
        pass

    print("Uploaded Firmware. Rebooting")


if __name__ == "__main__":
    if dev := find_usb(VENDOR_ID, BOOT_PRODUCT_ID):
        upload(dev, FIRMWARE_PATH)
