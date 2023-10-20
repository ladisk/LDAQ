import threading
import asyncio
import struct
from bleak import BleakClient
import numpy as np
import time

address = "B4:E6:2D:98:56:2B"  # Replace with your ESP32's device address
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

recieved_data = []

def callback(sender: int, data: bytearray):
    global recieved_data
    # Data received is in bytes, convert it to integers
    sine_wave1, sine_wave2, sine_wave3 = struct.unpack('iii', data)
    #print(f"Sine Wave 1: {sine_wave1}, Sine Wave 2: {sine_wave2}, Sine Wave 3: {sine_wave3}")
    recieved_data.append([sine_wave1, sine_wave2, sine_wave3])
    


async def main(duration):
    client = BleakClient(address)
    try:
        await client.connect()
        print(f"Connected: {client.is_connected}")
        services = await client.get_services()
        for service in services:
            if service.uuid == SERVICE_UUID:
                for char in service.characteristics:
                    if char.uuid == CHARACTERISTIC_UUID:
                        await client.start_notify(char, callback)

        time_start = time.time()
        while True:
            await asyncio.sleep(1)  # Keep the event loop alive by sleeping
            
            if time.time() - time_start >= duration:
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if client.is_connected:
            await client.disconnect()
            print(f"Disconnected: {not client.is_connected}")

def run_ble_in_thread(duration):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(duration))
    finally:
        loop.close()

def acquire(duration=10):
    global recieved_data
    recieved_data = []
    
    # start the thread
    thread = threading.Thread(target=run_ble_in_thread, args=(duration,) )
    thread.start()

    print(f"Waiting for {duration} seconds...")
    thread.join()
    print("Ended")
    return np.array(recieved_data)


# import asyncio
# import struct
# from bleak import BleakClient

# address = "B4:E6:2D:98:56:2B"  # Replace with your ESP32's device address
# SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
# CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# def callback(sender: int, data: bytearray):
#     # Data received is in bytes, convert it to integers
#     sine_wave1, sine_wave2, sine_wave3 = struct.unpack('iii', data)
#     print(f"Sine Wave 1: {sine_wave1}, Sine Wave 2: {sine_wave2}, Sine Wave 3: {sine_wave3}")

# async def main():
#     async with BleakClient(address) as client:
#         print(f"Connected: {client.is_connected}")

#         services = await client.get_services()
#         for service in services:
#             if service.uuid == SERVICE_UUID:
#                 for char in service.characteristics:
#                     if char.uuid == CHARACTERISTIC_UUID:
#                         await client.start_notify(char, callback)

#         while True:
#             await asyncio.sleep(1)  # Keep the event loop alive by sleeping

# # Get the default event loop and run the main function until complete
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())

