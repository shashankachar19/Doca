import os
import time
import pyautogui
import subprocess
import signal

def main():
    print("Starting ui_automation.py...")
    # Give the app a moment to launch and render
    time.sleep(3)

    print("Clicking 'Database Dashboard' tab...")
    # Coordinates might need tweaking, assuming default window size and position
    # The third tab is usually somewhat offset. Let's send keystrokes or click a relative position.
    # To be safe across Xvfb screens, let's use keyboard navigation if possible.
    # Qt Tab widgets can often be navigated with Ctrl+Tab

    # Try clicking the tab directly (approximate position based on 1100x750 size, tabs are at the top)
    # The window title is "DoCA \u2014 Document Classification and Analysis"

    # Switch to Database tab (index 2)
    pyautogui.hotkey('ctrl', 'tab')
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'tab')
    time.sleep(1)

    print("Clicking 'Refresh Data'...")
    # Tab into the "Refresh Data" button and press Space, or just click it
    # We can try to hit Tab a few times and space, or just click.
    # We are now in the Database tab. The first focusable widget might be the Refresh button.
    pyautogui.press('tab')
    time.sleep(0.5)
    pyautogui.press('space')
    time.sleep(3)

    print("Scrolling through the table...")
    # Click inside the table to focus it, then page down
    pyautogui.click(x=300, y=300) # Arbitrary point in the table body
    time.sleep(0.5)

    for _ in range(5):
        pyautogui.press('pagedown')
        time.sleep(1)

    print("Done. Sending SIGINT to ffmpeg and closing app.")

    # Try to close the app via Ctrl+Q as defined in main_dashboard.py
    pyautogui.hotkey('ctrl', 'q')
    time.sleep(1)

if __name__ == '__main__':
    main()
