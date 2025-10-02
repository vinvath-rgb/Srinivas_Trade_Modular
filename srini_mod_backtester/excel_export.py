import io, pandas as pd

def to_excel(inputs: dict, stats: dict, equity: pd.Series) -> bytes:
    with io.BytesIO() as out:
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            pd.DataFrame(list(inputs.items()), columns=["Parameter","Value"]).to_excel(w, "Inputs", index=False)
            pd.DataFrame([stats]).to_excel(w, "Stats", index=False)
            equity.rename("Equity").to_frame().to_excel(w, "Equity", index=True)
        return out.getvalue()
