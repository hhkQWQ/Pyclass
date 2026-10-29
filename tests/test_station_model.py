"""站点步骤独立契约测试：已知端点、冻结筛选、共享拟合和整周边界。"""
import unittest

import numpy as np
import pandas as pd

from src.station_model import (
    extract_events, select_stations, build_panel, before, design,
    fit_shared, run_experiment, daily_weekly_totals, summary,
    PHASES, MONTH_START, MONTH_END, SELECTION_START,
)


def trip_frame():
    started = pd.date_range('2024-01-01 12:00', periods=31, freq='D')
    data = pd.DataFrame({'ride_id':[str(i) for i in range(31)], 'started_at':started,
                         'ended_at':started+pd.Timedelta(minutes=5), 'duration_min':5.,
                         'start_station_id':'A', 'end_station_id':'B',
                         'start_station_name':'Alpha', 'end_station_name':'Beta'})
    data['hour']=started.hour;data['weekday']=started.dayofweek;data['is_weekend']=started.dayofweek>=5
    return data


def event(station, timestamp, direction, ride_id=None):
    timestamp=pd.Timestamp(timestamp)
    return {'ride_id':ride_id or f'{station}/{timestamp}/{direction}', 'station_id':station,
            'station_name':station, 'event_at':timestamp, 'timestamp':timestamp.floor('h'), 'direction':direction}


def synthetic_panel(station_count=2):
    ids=[f'S{i}' for i in range(station_count)]
    index=pd.date_range(MONTH_START,MONTH_END,inclusive='left',freq='h')
    multi=pd.MultiIndex.from_product([ids,index],names=['station_id','timestamp'])
    parts=[]
    for number in range(station_count):
        out=4+number+index.hour.to_numpy()%6+index.dayofweek.to_numpy()%3
        inflow=3+number+(index.hour.to_numpy()+2)%6+index.dayofweek.to_numpy()%2
        part=pd.DataFrame({'outflow':out.astype(float),'inflow':inflow.astype(float)},index=index)
        part.loc[part.index<SELECTION_START,'inflow']=np.nan
        part['net_outflow']=part.outflow-part.inflow
        part['inflow_complete']=part.index>=SELECTION_START
        parts.append(part)
    panel=pd.concat(parts);panel.index=multi
    selected=pd.DataFrame({'station_id':ids,'station_name':ids,'selection_rank':range(1,station_count+1)})
    return panel,selected


SMALL_GRIDS={'historical_mean':[{}],'last_week':[{}],'ridge':[{'alpha':1.}],
             'random_forest':[{'n_estimators':3,'max_depth':3,'min_samples_leaf':8,'random_state':42,'n_jobs':1}]}


class StationPredictionTests(unittest.TestCase):
    def test_independent_endpoint_retention(self):
        data=trip_frame()
        data.loc[0,'end_station_id']=pd.NA
        data.loc[1,'start_station_id']=pd.NA
        data.loc[2,['start_station_id','end_station_id']]=pd.NA
        events,info=extract_events(data)
        self.assertEqual(len(events),58)
        self.assertEqual(info['only_start_known'],1)
        self.assertEqual(info['only_end_known'],1)
        self.assertEqual(info['neither_endpoint_known'],1)
        self.assertEqual(events.loc[events.ride_id.eq('0'),'direction'].tolist(),['outflow'])
        self.assertEqual(events.loc[events.ride_id.eq('1'),'direction'].tolist(),['inflow'])

    def test_cross_origin_return_not_counted_early(self):
        data=trip_frame()
        row=data.index[13]
        data.loc[row,'started_at']=pd.Timestamp('2024-01-14 23:55')
        data.loc[row,'ended_at']=pd.Timestamp('2024-01-15 00:05')
        data.loc[row,'duration_min']=10.
        data.loc[row,'hour']=23
        events,_=extract_events(data)
        chosen=events.loc[events.ride_id.eq(str(row))]
        historical=chosen.loc[chosen.event_at<pd.Timestamp('2024-01-15')]
        self.assertEqual(historical.direction.tolist(),['outflow'])
        self.assertEqual(chosen.loc[chosen.direction.eq('inflow'),'timestamp'].iloc[0],pd.Timestamp('2024-01-15'))

    def test_tidal_score_does_not_cancel_morning_and_evening(self):
        # 每天A在08时借出、18时归还，B完全相反；全天净流出都为0。
        events=[]
        for day in pd.date_range('2024-01-01',periods=14,freq='D'):
            for station,first,second in [('A','outflow','inflow'),('B','inflow','outflow')]:
                events.extend([event(station,day+pd.Timedelta(hours=8),first),
                               event(station,day+pd.Timedelta(hours=18),second)])
        candidates,chosen=select_stations(pd.DataFrame(events),top_n=2,min_daily_activity=0,min_active_days=1)
        self.assertEqual(chosen.station_id.tolist(),['A','B'])
        np.testing.assert_allclose(chosen.tidal_score.to_numpy(),[2.,2.])
        self.assertTrue((candidates.daily_activity==2).all())

    def test_future_events_and_names_cannot_change_frozen_selection(self):
        events=[]
        for day in pd.date_range('2024-01-01',periods=14,freq='D'):
            events += [event('A',day+pd.Timedelta(hours=8),'outflow'),event('B',day+pd.Timedelta(hours=9),'inflow')]
        current=pd.DataFrame(events)
        _,selected=select_stations(current,top_n=2,min_daily_activity=0,min_active_days=1)
        extra=pd.DataFrame([event('C','2024-01-20 08:00','outflow',str(i)) for i in range(1000)])
        changed=pd.concat([current,extra],ignore_index=True)
        _,selected_again=select_stations(changed,top_n=2,min_daily_activity=0,min_active_days=1)
        pd.testing.assert_frame_equal(selected,selected_again)

    def test_insufficient_candidates_do_not_lower_thresholds(self):
        with self.assertRaisesRegex(ValueError,'不足'):
            select_stations(pd.DataFrame([event('A','2024-01-02 08:00','outflow')]))

    def test_panel_boundary_unknown_returns_and_february_exclusion(self):
        events=pd.DataFrame([event('A','2024-01-01 00:00','outflow'),event('A','2024-01-01 00:00','inflow'),
                             event('A','2024-01-01 03:00','inflow'),event('A','2024-02-01 00:00','inflow')])
        panel=build_panel(events,['A'])
        self.assertEqual(len(panel),744)
        self.assertEqual(panel.inflow.isna().sum(),3)
        self.assertEqual(panel.outflow.sum(),1)
        self.assertEqual(panel.inflow.sum(),1)
        self.assertTrue(np.isnan(panel.loc[('A',pd.Timestamp('2024-01-01 00:00')),'net_outflow']))
        self.assertEqual(panel.loc[('A',pd.Timestamp('2024-01-01 03:00')),'net_outflow'],-1)

    def test_shared_row_counts_categories_and_training_scaler(self):
        panel,selected=synthetic_panel()
        ids=selected.station_id.tolist()
        history=before(panel,pd.Timestamp('2024-01-15'))
        x,y=design(history,ids,'outflow')
        self.assertEqual(x.shape,(336,17))
        self.assertEqual(set(x.index.get_level_values('station_id')),set(ids))
        self.assertTrue((x.xs('S0').station_1==0).all())
        self.assertTrue((x.xs('S1').station_1==1).all())
        fitted=fit_shared('ridge','outflow',history,ids,{'alpha':1.})
        np.testing.assert_allclose(fitted.estimator.named_steps['standardscaler'].mean_,x.mean().to_numpy())
        self.assertEqual(fitted.fit_rows,336)
        self.assertEqual(fit_shared('ridge','inflow',history,ids,{'alpha':1.}).fit_rows,330)
        test_history=before(panel,pd.Timestamp('2024-01-22'))
        self.assertEqual(fit_shared('ridge','inflow',test_history,ids,{'alpha':1.}).fit_rows,666)

    def test_shared_baselines_match_station_specific_hand_values(self):
        panel,selected=synthetic_panel();ids=selected.station_id.tolist();origin=pd.Timestamp('2024-01-15')
        history=before(panel,origin)
        predicted=fit_shared('historical_mean','inflow',history,ids,{}).predict(origin)
        stamp=pd.Timestamp('2024-01-15 00:00')
        # 月初这一小时归还未知，因此同期均值只能用Jan8的已知值。
        self.assertEqual(predicted.loc[('S1',stamp),'prediction'],panel.loc[('S1',pd.Timestamp('2024-01-08')),'inflow'])
        seasonal=fit_shared('last_week','outflow',history,ids,{}).predict(origin)
        self.assertEqual(seasonal.loc[('S0',stamp),'prediction'],panel.loc[('S0',pd.Timestamp('2024-01-08')),'outflow'])
        with self.assertRaisesRegex(ValueError,'预测起点之后'):
            fit_shared('ridge','outflow',panel,ids,{'alpha':1.}).predict(origin)

    def test_future_test_labels_do_not_change_recommendation_or_predictions(self):
        panel,selected=synthetic_panel()
        first=run_experiment(panel,selected,SMALL_GRIDS)
        changed=panel.copy();future=changed.index.get_level_values('timestamp')>=pd.Timestamp('2024-01-22')
        changed.loc[future,'outflow']+=9999;changed.loc[future,'inflow']+=5555
        changed['net_outflow']=changed.outflow-changed.inflow
        second=run_experiment(changed,selected,SMALL_GRIDS)
        self.assertEqual(first.selection,second.selection)
        pd.testing.assert_frame_equal(first.candidates,second.candidates)
        np.testing.assert_array_equal(first.predictions.prediction,second.predictions.prediction)
        self.assertFalse(first.predictions.actual.equals(second.predictions.actual))
        self.assertNotEqual(first.selection['recommended_model'],'last_week')

    def test_net_is_difference_of_clipped_counts_and_can_be_negative(self):
        panel,selected=synthetic_panel()
        result=run_experiment(panel,selected,SMALL_GRIDS)
        pivot=result.predictions.pivot(index=['phase','station_id','timestamp','model'],columns='target',values='prediction')
        np.testing.assert_allclose(pivot.net_outflow,pivot.outflow-pivot.inflow)
        self.assertTrue((pivot.net_outflow<0).any())
        self.assertTrue((pivot.outflow>=0).all() and (pivot.inflow>=0).all())
        totals=daily_weekly_totals(result.predictions)
        self.assertTrue(totals.loc[totals.target.eq('net_outflow'),'absolute_percentage_error'].isna().all())


if __name__=='__main__':
    unittest.main(verbosity=2)
