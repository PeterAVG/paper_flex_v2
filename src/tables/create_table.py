import pandas as pd

from src.base import Case
from src.evaluation import average_opt_results
from src.experiment_manager.cache import load_cache


def create_latex_tables_analysis1() -> None:

    ### TABLE 1: compare first 3 cases: SPOT vs mFRR_AND_ENERGY vs mFRR ###
    cache = load_cache()
    res = []
    # get all files in folder
    # r = [results for _params, results in cache.items()]
    for _params, results in cache.items():
        params = eval(_params)
        if "analysis1" == params["analysis"] and (
            (
                Case.SPOT.name in params["case"]
                or Case.mFRR_AND_ENERGY.name in params["case"]
            )
            # or Case.mFRR.name in params["case"]
            and not params.get("gamma", False)
            and not params.get("one_lambda", False)
        ):
            opt_result = results[1]
            if isinstance(opt_result, list) and len(opt_result) > 1:
                opt_result = average_opt_results(opt_result)
            year = params["year"]
            name = params["case"]
            admm = params.get("admm", False)

            name = name.replace(".", " ")
            one_lambda = "One bid" if params["one_lambda"] else ""
            name = (
                name + " " + one_lambda
                if Case.mFRR_AND_ENERGY.name in params["case"]
                else name
            )

            nb = params.get("nb_scenarios_spot", 1)
            print(nb)

            if nb > 0 or nb == -4:
                _nb = "All" if nb == -4 else nb
                res.append(
                    {
                        **{"Name": name},
                        **{"Year": year},
                        **opt_result.__dict__,
                        **{"Scenarios": _nb},
                        **{"ADMM": admm},
                    }
                )
                # print(f"\n{name}: \n{opt_result.__repr__()}")
        elif "analysis3" == params["analysis"] and (
            Case.SPOT.name in params["case"]
            or Case.mFRR_AND_ENERGY.name in params["case"]
            # or Case.mFRR.name in params["case"]
            and not params.get("gamma", False)
            and not params.get("one_lambda", False)
        ):
            opt_result = average_opt_results(results[1])
            name = "Receding horizon"
            one_lambda = "One bid" if params["one_lambda"] else ""
            name = (
                name + " " + one_lambda
                if Case.mFRR_AND_ENERGY.name in params["case"]
                else name
            )
            lookback = params["lookback"]
            year = params["year"]
            admm = params.get("admm", False)

            res.append(
                {
                    **{"Name": name},
                    **{"Year": year},
                    # **{"Lookback": lookback},
                    **opt_result.__dict__,
                    **{"Scenarios": -lookback},
                    **{"ADMM": admm},
                }
            )

    # create dataframe of all results in res. Each element is of type OptimizationResult
    df = pd.DataFrame(res).round(3)
    df.columns = [c.replace("_", " ").capitalize() for c in df.columns]
    df = df.sort_values(["Year", "Name"]).set_index(["Year", "Name"]).T
    df.drop(["Battery capacity"], inplace=True)

    df.loc["% savings"] = (
        (1 - df.loc["Total cost"].astype(float) / df.loc["Base cost today"])
        .astype(float)
        .round(3)
        .values
    ) * 100
    print(df[2021])
    print(df[2022])
    df_2022 = df[2022]
    df_2022.loc["Name", :] = df_2022.columns.tolist()
    table_compare_3 = df_2022.T.query(
        "(Name == 'SPOT') | (Scenarios == -5) | (Scenarios == 50)"
    ).T
    table_compare_3.rename({Case.mFRR_AND_ENERGY.name: "mFRR"}, inplace=True)
    table_compare_3.drop(["Name"], inplace=True)
    table_compare_3.rename(
        columns={
            "Receding horizon ": "mFRR w. lookback",
            "SPOT": "Load shifting",
            "mFRR_AND_ENERGY ": "mFRR w. 2021",
        },
        inplace=True,
    )
    print(table_compare_3)
    print(table_compare_3)
    pass
    # # print df to latex
    align_str = "".join(["c" for _ in range(len(table_compare_3.columns))])
    print(
        table_compare_3.to_latex(multicolumn_format="c", column_format=f"l{align_str}")
    )

    # ## TABLE 2: compare next 3 cases: naive vs mFRR_AND_ENERGY vs robust ###
    # res = []
    # for _params, results in cache.items():
    #     params = eval(_params)

    #     if "analysis1" == params["analysis"] and (
    #         Case.NAIVE.name in params["case"]
    #         or Case.ROBUST.name in params["case"]
    #         or Case.mFRR_AND_ENERGY.name in params["case"]
    #     ):
    #         opt_result = results[1]
    #         year = params["year"]
    #         name = params["case"]
    #         name = name.replace(".", " ")
    #         one_lambda = "One bid" if params["one_lambda"] else ""
    #         name = (
    #             name + " " + one_lambda
    #             if (
    #                 Case.mFRR_AND_ENERGY.name in params["case"]
    #                 or Case.ROBUST.name in name
    #             )
    #             else name
    #         )

    #         res.append({**{"Name": name}, **{"Year": year}, **opt_result.__dict__})
    #         print(f"\n{name}: \n{opt_result.__repr__()}")

    # # create dataframe of all results in res. Each element is of type OptimizationResult
    # df = pd.DataFrame(res).round(1)
    # df.columns = [c.replace("_", " ").capitalize() for c in df.columns]
    # df = df.sort_values(["Year", "Name"]).set_index(["Year", "Name"]).T
    # df.drop(["Battery capacity"], inplace=True)

    # df.loc["% savings"] = (
    #     (1 - df.loc["Total cost"] / df.loc["Base cost today"]).round(2).values
    # )
    # print(df)

    # # print df to latex
    # align_str = "".join(["c" for _ in range(len(df.columns))])
    # print(df.to_latex(multicolumn_format="c", column_format=f"l{align_str}"))

    pass


if __name__ == "__main__":
    create_latex_tables_analysis1()
