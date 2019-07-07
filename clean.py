def get_filter_by_row(input, columns):
    filter_data = {}
    for col in columns:
        if "columns" in col:
            for df_column in col["columns"]:
                if input[df_column] == 1:
                    filter_data[df_column] = 1
        elif "name" in col:
            filter_data[col["name"]] = input[col["name"]]
    return filter_data
