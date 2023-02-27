def generate_step_features(df, max_d=20, prefix='step'):
    xx_wgts = []
    for d in range(1, max_d + 1):
        wname = f'{prefix}_{d}'
        xx = df[['session', 'aid']].copy()
        xx['session_shift'] = xx.session.shift(-d)
        xx['aid2'] = xx.aid.shift(-d, fill_value=0)
        xx = xx[['aid', 'aid2']][xx.session == xx.session_shift]
        xx[wname] = 1
        xx = xx.groupby(['aid', 'aid2']).count()
        xx.sort_index(inplace=True)
        xx_wgts.append(xx)

    # faster binary tree joined
    for i in range(0, len(xx_wgts) * 2 - 2, 2):
        new_f = xx_wgts[i].join(xx_wgts[i + 1], how='outer')
        xx_wgts.append(new_f)
    res = xx_wgts[-1]

    res.fillna(0, inplace=True)
    res.reset_index(inplace=True)
    res = res.astype(int)
    return res