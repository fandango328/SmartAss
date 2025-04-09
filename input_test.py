#!/usr/bin/env python3
from evdev import InputDevice, categorize, ecodes, list_devices
import select
import time
import os

print(f"Current user: {os.getlogin()}")
print(f"Current UID: {os.getuid()}")
print(f"Current GID: {os.getgid()}")
print(f"Current groups: {os.getgroups()}")

print("\nAvailable input devices:")
for path in list_devices():
    try:
        device = InputDevice(path)
        print(f"Device: {device.name} at {path}")
        # Check if we can read from this device
        try:
            select.select([device.fd], [], [], 0)
            print(f"  - Can read from device: YES")
        except Exception as e:
            print(f"  - Can read from device: NO - {e}")
            
        # Check if we can get exclusive access
        try:
            device.grab()
            print(f"  - Can grab device: YES")
            device.ungrab()
        except Exception as e:
            print(f"  - Can grab device: NO - {e}")
            
    except Exception as e:
        print(f"Error with device {path}: {e}")

print("\nListening for key events for 10 seconds... Press keys (especially LEFTMETA/Windows key)")
print("Press Ctrl+C to exit early\n")

# Try to find the Pi 500 keyboard
keyboard = None
for path in list_devices():
    try:
        device = InputDevice(path)
        if "Pi 500" in device.name and "Keyboard" in device.name and "Mouse" not in device.name:
            print(f"Using keyboard: {device.name} at {device.path}")
            keyboard = device
            break
    except Exception as e:
        pass

if not keyboard:
    print("No Pi 500 keyboard found!")
    exit(1)

try:
    start_time = time.time()
    while time.time() - start_time < 10:
        r, w, x = select.select([keyboard.fd], [], [], 0.1)
        if r:
            try:
                for event in keyboard.read():
                    if event.type == ecodes.EV_KEY:
                        key_name = "UNKNOWN"
                        try:
                            key_name = ecodes.KEY[event.code]
                            if isinstance(key_name, tuple):
                                key_name = key_name[0]
                        except:
                            pass
                        state = "PRESSED" if event.value == 1 else "RELEASED" if event.value == 0 else "REPEATED"
                        print(f"Key: {key_name} (Code: {event.code}) - {state}")
            except Exception as e:
                print(f"Error reading events: {e}")
except KeyboardInterrupt:
    print("\nExiting key monitor")