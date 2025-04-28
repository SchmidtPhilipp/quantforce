import subprocess

def bring_safari_tab_to_front(title_match="TensorBoard", url_match="http://localhost:6006"):
    applescript = f'''
    tell application "Safari"
        set target_url to "{url_match}"
        set target_title to "{title_match}"
        set found to false

        repeat with w in windows
            set i to 1
            repeat with t in tabs of w
                if (URL of t contains target_url or name of t contains target_title) then
                    set current tab of w to t
                    set index of w to 1
                    activate
                    set found to true
                    exit repeat
                end if
                set i to i + 1
            end repeat
            if found then exit repeat
        end repeat

        if not found then
            set new_tab to make new document with properties {{URL:target_url}}
            activate
        end if
    end tell
    '''
    subprocess.run(["osascript", "-e", applescript])


def focus_tensorboard_tab(title_match="TensorBoard", url_match="http://localhost:6006"):
    applescript = f'''
    tell application "Safari"
        set target_url to "{url_match}"
        set target_title to "{title_match}"
        repeat with w in windows
            repeat with t in tabs of w
                if (URL of t contains target_url or name of t contains target_title) then
                    set current tab of w to t
                    set index of w to 1
                    activate
                    return
                end if
            end repeat
        end repeat
    end tell
    '''
    subprocess.run(["osascript", "-e", applescript])


def refresh_tensorboard_tab(title_match="TensorBoard", url_match="http://localhost:6006"):
    applescript = f'''
    tell application "Safari"
        set target_url to "{url_match}"
        set target_title to "{title_match}"
        repeat with w in windows
            repeat with t in tabs of w
                if (URL of t contains target_url or name of t contains target_title) then
                    set t's URL to target_url
                    return
                end if
            end repeat
        end repeat
    end tell
    '''
    subprocess.run(["osascript", "-e", applescript])


def refresh_current_safari_window():
    """
    Refreshes the current active window in Safari.
    """
    applescript = '''
    tell application "Safari"
        if (count of windows) > 0 then
            tell front window
                set current_tab to current tab
                set current_tab's URL to (current_tab's URL) -- Refresh the current tab
            end tell
        end if
    end tell
    '''
    subprocess.run(["osascript", "-e", applescript])
