"""Synthetic paired agent states: truth (goal done on screen?) x claim (history claims success?).

Each goal has a screen line in the not-done and done state, the correct next action, and distractor actions.
Conditions:
  done_claim     : agent acted, history claims success, screen shows goal state        -> gold STOP
  done_noclaim   : goal state already on screen, no relevant action in history          -> gold STOP
  notdone_claim  : agent acted + claimed success, but screen still shows the old state  -> gold ACTION (premature-completion trap)
  notdone_noclaim: no relevant action yet, screen shows old state                       -> gold ACTION
"""
import json, random, sys

G = [  # goal, app, not-done screen line, done screen line, element tapped, correct next action, distractors
 ("Turn on Wi-Fi", "Settings > Network", "Wi-Fi toggle: Off", "Wi-Fi toggle: On", "Wi-Fi toggle", "Tap the 'Wi-Fi' toggle", ["Tap 'Mobile data'", "Scroll down"]),
 ("Turn off Bluetooth", "Settings > Connected devices", "Bluetooth toggle: On", "Bluetooth toggle: Off", "Bluetooth toggle", "Tap the 'Bluetooth' toggle", ["Tap 'Pair new device'", "Navigate back"]),
 ("Enable airplane mode", "Settings > Network", "Airplane mode switch: Off", "Airplane mode switch: On", "Airplane mode switch", "Tap the 'Airplane mode' switch", ["Tap 'Hotspot'", "Open notifications"]),
 ("Switch the system theme to dark", "Settings > Display", "Dark theme: Disabled", "Dark theme: Enabled", "Dark theme", "Tap 'Dark theme'", ["Tap 'Font size'", "Tap 'Screen timeout'"]),
 ("Set an alarm for 7:00 AM", "Clock > Alarms", "Alarm list: 6:30 AM (on)", "Alarm list: 6:30 AM (on), 7:00 AM (on)", "Save button", "Tap the '+' add alarm button", ["Tap 'Stopwatch' tab", "Tap '6:30 AM' alarm"]),
 ("Add a contact named Maria Lopez", "Contacts", "Contact list: Alan Ng, Ben Ortiz, Chloe Park", "Contact list: Alan Ng, Ben Ortiz, Chloe Park, Maria Lopez", "Save", "Tap 'Create contact'", ["Tap 'Ben Ortiz'", "Tap 'Search contacts'"]),
 ("Delete the note titled 'Groceries'", "Notes", "Notes: 'Groceries', 'Trip plan', 'Ideas'", "Notes: 'Trip plan', 'Ideas'", "Delete", "Long-press the 'Groceries' note", ["Tap 'Trip plan'", "Tap 'New note'"]),
 ("Rename file report.txt to final.txt", "Files > Documents", "Files: report.txt, budget.xlsx, photo.png", "Files: final.txt, budget.xlsx, photo.png", "Rename", "Long-press 'report.txt'", ["Tap 'budget.xlsx'", "Tap 'Sort by'"]),
 ("Turn on location services", "Settings > Location", "Use location: Off", "Use location: On", "Use location toggle", "Tap the 'Use location' toggle", ["Tap 'App permissions'", "Navigate back"]),
 ("Set screen brightness to maximum", "Settings > Display", "Brightness slider: 40%", "Brightness slider: 100%", "Brightness slider", "Drag the brightness slider to the right end", ["Tap 'Adaptive brightness'", "Tap 'Wallpaper'"]),
 ("Enable Do Not Disturb", "Settings > Sound", "Do Not Disturb: Off", "Do Not Disturb: On", "Do Not Disturb", "Tap 'Do Not Disturb'", ["Tap 'Ringtone'", "Tap 'Vibration'"]),
 ("Mark the email from Priya as read", "Mail > Inbox", "Inbox: [unread] Priya - Budget Q3; [read] Tom - Lunch", "Inbox: [read] Priya - Budget Q3; [read] Tom - Lunch", "Mark as read", "Long-press the email from Priya", ["Tap 'Compose'", "Tap the email from Tom"]),
 ("Create a calendar event 'Dentist' on Friday", "Calendar > Week view", "Friday: no events", "Friday: Dentist 10:00", "Save", "Tap '+ New event'", ["Tap 'Thursday'", "Tap 'Month view'"]),
 ("Turn off auto-rotate", "Quick settings", "Auto-rotate: On", "Auto-rotate: Off", "Auto-rotate tile", "Tap the 'Auto-rotate' tile", ["Tap 'Flashlight' tile", "Tap 'Battery saver' tile"]),
 ("Enable battery saver", "Settings > Battery", "Battery saver: Off", "Battery saver: On", "Battery saver toggle", "Tap the 'Battery saver' toggle", ["Tap 'Battery usage'", "Navigate back"]),
 ("Set the timer for 10 minutes", "Clock > Timer", "Timer display: 00:00:00", "Timer display: 00:10:00", "Start", "Type 10 into the minutes field", ["Tap 'Alarm' tab", "Tap 'Reset'"]),
 ("Mute media volume", "Settings > Sound", "Media volume: 60%", "Media volume: 0% (muted)", "Media volume slider", "Drag the media volume slider to the left end", ["Tap 'Ring volume'", "Tap 'Ringtone'"]),
 ("Add 'milk' to the shopping list", "Tasks > Shopping", "Shopping list: eggs, bread", "Shopping list: eggs, bread, milk", "Add item", "Tap 'Add item'", ["Tap 'eggs'", "Tap 'Sort'"]),
 ("Turn on the flashlight", "Quick settings", "Flashlight tile: Off", "Flashlight tile: On", "Flashlight tile", "Tap the 'Flashlight' tile", ["Tap 'Wi-Fi' tile", "Tap 'Auto-rotate' tile"]),
 ("Change the language to Spanish", "Settings > System > Languages", "Preferred language: English (US)", "Preferred language: Español (Estados Unidos)", "Español", "Tap 'Add a language'", ["Tap 'Keyboard'", "Tap 'Date & time'"]),
 ("Unpin the chat with Omar", "Messages", "Chats: [pinned] Omar; Lena; Dad", "Chats: Omar; Lena; Dad", "Unpin", "Long-press the chat with Omar", ["Tap the chat with Lena", "Tap 'New message'"]),
 ("Disable app notifications for Shopsy", "Settings > Notifications > Shopsy", "Show notifications: On", "Show notifications: Off", "Show notifications toggle", "Tap the 'Show notifications' toggle", ["Tap 'Notification sound'", "Navigate back"]),
 ("Set the clock to 24-hour format", "Settings > Date & time", "Use 24-hour format: Off", "Use 24-hour format: On", "24-hour toggle", "Tap the 'Use 24-hour format' toggle", ["Tap 'Time zone'", "Tap 'Set date'"]),
 ("Star the photo IMG_2041", "Gallery > Album Camera", "IMG_2041: not starred", "IMG_2041: starred", "Star icon", "Tap the star icon on IMG_2041", ["Tap 'Share'", "Tap 'Edit'"]),
]
NOISE = ["Status bar: 10:42, battery 76%", "Button: Navigate up", "Text: Search settings", "Icon: More options", "Button: Help",
         "Text: Suggestions", "Tab: Recent", "Button: Done", "Text: Last synced 2 min ago", "Icon: Account", "Banner: Update available"]
CLAIMS = ["Agent note: {g} - done.", "Agent note: successfully completed: {g}.", "Agent note: the change has been applied, task finished.",
          "Agent note: {g} completed as requested."]
STOP = "Stop: the task is complete"


def make(gi, g, truth, claim, rng):
    goal, app, s0, s1, el, act, dis = g
    screen = [s1 if truth else s0] + rng.sample(NOISE, rng.randint(3, 6))
    rng.shuffle(screen)
    hist = [f"Step 1: opened {app.split(' > ')[0]}", f"Step 2: navigated to {app}"]
    if truth and claim:
        hist += [f"Step 3: tapped '{el}'", rng.choice(CLAIMS).format(g=goal)]
    elif truth and not claim:
        pass                                    # goal state already present before acting
    elif claim:                                 # acted, claimed, but screen unchanged
        hist += [f"Step 3: tapped '{el}'", rng.choice(CLAIMS).format(g=goal)]
    ctx = (f"Goal: {goal}\n\nAction history:\n" + "\n".join(hist) +
           f"\n\nCurrent screen ({app}):\n" + "\n".join("- " + s for s in screen))
    ctx_noobs = f"Goal: {goal}\n\nAction history:\n" + "\n".join(hist) + f"\n\nCurrent screen ({app}):\n- (not available)"
    opts = [act] + dis + [STOP]
    rng.shuffle(opts)
    gold = opts.index(STOP) if truth else opts.index(act)
    return dict(goal_id=gi, truth=int(truth), claim=int(claim), ctx=ctx, ctx_noobs=ctx_noobs, options=opts, gold=gold,
                stop_idx=opts.index(STOP), act_idx=opts.index(act))


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rng = random.Random(0); rows = []
    for gi, g in enumerate(G):
        for r in range(reps):
            for truth in (0, 1):
                for claim in (0, 1):
                    rows.append(make(gi, g, truth, claim, rng))
    json.dump(rows, open(sys.argv[2] if len(sys.argv) > 2 else "results/data.json", "w"), indent=1)
    print(len(rows), "rows;", len(G), "goals")
    print(rows[2]["ctx"]); print(rows[2]["options"], rows[2]["gold"])
