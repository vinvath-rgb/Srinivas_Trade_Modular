# streamlit_app.py
from srini_mod_backtester.run import main

if __name__ == "__main__":
    main()
else:
    # Streamlit runs as a module; still call main once
    main()