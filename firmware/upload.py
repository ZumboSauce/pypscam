import usb

OMNIVISION_VID = 0x05A9
OMNIVISION_BOOT_PID = 0x0580

def upload( path: str ):
    with open( path, "rb" ) as fw:
        if devices := usb.core.find( idVendor=OMNIVISION_VID, idProduct=OMNIVISION_BOOT_PID, find_all=True ):
            devices = list(( dev for dev in devices if isinstance( dev, usb.core.Device ) ))
            print( f"Found {len(devices)} targets for upload" )
            for index, dev in enumerate( devices, start = 1 ):
                try:
                    print( f"Trying firmware upload to camera {index}" )
                    dev.set_configuration()
                    page = 0x14 #Borrowed this from psxdev's upload script in OrbisEyeCam. Thank you !
                    addr = 0 #Again
                    while chunk := fw.read(512):
                        dev.ctrl_transfer(0x40, 0x0, addr, page, chunk)
                        if (addr := addr + len(chunk)) > 0xFFFF:
                            addr = 0
                            page += 1
                    print( f"Firmware upload to camera {index} successful" )
                except Exception as e:
                    print( f"Firmware upload to camera {index} failed: ", e )
                    print( "Skipping to next camera" )
