"""Shared request definitions for the overhead benchmarks."""
OPS_Q = {"type": "choice", "instructions": "Following native_t3a_prompt, choose the next single action type to advance the goal.",
         "criteria": {k: v for k, v in [('click', 'Click a UI element.'), ('long_press', 'Long press a UI element.'),
         ('input_text', 'Input text in a UI element.'), ('scroll', 'Scroll a UI element or screen.'), ('open_app', 'Open an installed app.'),
         ('keyboard_enter', 'Press Enter.'), ('navigate_back', 'Go back.'), ('navigate_home', 'Go home.'), ('wait', 'Wait for an update.'),
         ('answer', 'Answer the user explicitly using observed information.'), ('complete', 'The requested goal is already achieved.'),
         ('infeasible', 'The goal cannot be achieved in this environment.')]}}
