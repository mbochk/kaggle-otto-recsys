def check_inclusion(df):
    ll = df[df.type == 'clicks'][['session', 'aid']].values.tolist()
    ss_clicks = set(map(tuple, ll))

    ll = df[df.type == 'carts'][['session', 'aid']].values.tolist()
    ss_carts = set(map(tuple, ll))
    carts_without_click = len(ss_carts.difference(ss_clicks))

    ll = df[df.type == 'orders'][['session', 'aid']].values.tolist()
    ss_orders = set(map(tuple, ll))
    orders_without_click = len(ss_orders.difference(ss_clicks))
    orders_without_carts = len(ss_orders.difference(ss_carts))

    print(f"Clicks total {ss_clicks}")
    # carts_without_click <10 %
    print(f"Carts total {ss_clicks}, {carts_without_click} without click")
    # both around 14%
    print(f"Orders total {ss_clicks}, {orders_without_click} without click, {orders_without_carts} withou cart")
