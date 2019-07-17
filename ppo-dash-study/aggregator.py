import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.summary import v1 as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2


# def tabulate_events(dpath):
def tabulate_events(paths):

    # summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]
    summary_iterators = [EventAccumulator(dname).Reload() for dname in paths]

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)

    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            # assert len(set(e.step for e in events)) == 1

            for e in events:
                t = (e.step, e.value)
                out[tag].append(t)

    # merge tags
    import itertools
    import operator
    def find_mean(l):
        it = itertools.groupby(l, operator.itemgetter(0))
        # ans = []
        for key, subiter in it:
            values = list(item[1] for item in subiter)
            mean = np.array(values).mean()
            # ans.append((key, mean))
            yield key, mean
        # return ans

    out2 = defaultdict(list)
    for tag in tags:
        tuples = out[tag]
        tuples = find_mean(tuples)
        tuples = sorted(tuples, key=lambda t: t[0])
        out2[tag] = tuples
    return out2


def write_combined_events(dpath, d_combined, dname='combined'):
# ['reward', 'floor', 'reward.std', 'floor.std', 'steps', 'FPS', 'value_loss', 'action_loss_', 'dist_entropy_']
    fpath = os.path.join(dpath, dname)
    writer = tf.summary.FileWriter(fpath)

    tags, values = zip(*d_combined.items())

    # We only need to specify the layout once (instead of per step).
    layout_summary = summary_lib.custom_scalar_pb(
        layout_pb2.Layout(category=[
            layout_pb2.Category(
                title='losses',
                chart=[
                    # layout_pb2.Chart(
                    #     title='losses',
                    #     multiline=layout_pb2.MultilineChartContent(
                    #         tag=[r'loss(?!.*margin.*)'],)),
                    layout_pb2.Chart(
                        title='floor',
                        margin=layout_pb2.MarginChartContent(
                            series=[
                                layout_pb2.MarginChartContent.Series(
                                    value='floor/mean',
                                    lower='floor/std_lower',
                                    upper='floor/std_upper'
                                ),
                            ],)),
                    # layout_pb2.Chart(
                    #     title='floor',
                    #     margin=layout_pb2.MarginChartContent(
                    #         series=[
                    #             layout_pb2.MarginChartContent.Series(
                    #                 value='floor/mean/scalar_summary',
                    #                 lower='floor/std_lower/scalar_summary',
                    #                 upper='floor/std_upper/scalar_summary'
                    #             ),
                    #         ],)),
                ]),
            # layout_pb2.Category(
            #     title='trig functions',
            #     chart=[
            #         layout_pb2.Chart(
            #             title='wave trig functions',
            #             multiline=layout_pb2.MultilineChartContent(
            #                 tag=[
            #                     r'trigFunctions/cosine', r'trigFunctions/sine'
            #                 ],)),
            #         # The range of tangent is different. Give it its own chart.
            #         layout_pb2.Chart(
            #             title='tan',
            #             multiline=layout_pb2.MultilineChartContent(
            #                 tag=[r'trigFunctions/tangent'],)),
            #     ],
            #     # This category we care less about. Make it initially closed.
            #     closed=True),
        ]))
    writer.add_summary(layout_summary)

    floor = d_combined['floor']
    floor_std = d_combined['floor.std']
    # for tag, tuples in zip(tags, values):
        # for t in tuples:
    for i, t in enumerate(floor):
        step = t[0]
        mean = t[1]
        lower = mean-floor_std[i][1]
        upper = mean+floor_std[i][1]
        summary = tf.Summary(value=[tf.Summary.Value(tag='floor/mean', simple_value=mean)])
        writer.add_summary(summary, global_step=step)
        summary = tf.Summary(value=[tf.Summary.Value(tag='floor/std_lower', simple_value=lower)])
        writer.add_summary(summary, global_step=step)
        summary = tf.Summary(value=[tf.Summary.Value(tag='floor/std_upper', simple_value=upper)])
        writer.add_summary(summary, global_step=step)

    # for tag, tuples in zip(tags, values):
    #     for t in tuples:
    #         step = t[0]
    #         value = t[1]
    #         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    #         writer.add_summary(summary, global_step=step)
    writer.flush()


jobs = {
    '000_baseline': ['000_baseline-01', '000_baseline-02', '000_baseline-03'],
    '002_reduce_action_space': ['002_reduce_action_space-02', '002_reduce_action_space-03', '002_reduce_action_space-04'],
    '003_recurrent': ['003_recurrent-01'],
    '005_large_scale_hyperparms': ['005_large_scale_hyperparms-07', '005large_scale_hyperparms-08'],
    '006_reduced_frame_stack': ['006_reduced_frame_stack-01', '006_reduced_frame_stack-02', '006_reduced_frame_stack-03'],
    '007_reduced_action_space_and_frame_stack': 
        ['007_reduced_action_space_and_frame_stack-01', 
         '007_reduced_action_space_and_frame_stack-02', 
         '007_reduced_action_space_and_frame_stack-03'],
    '008_ra+rf+lshp': 
        ['008_ra+rf+lshp-01', 
         '008_ra+rf+lshp-02', 
         '008_ra+rf+lshp-03'],
    '009_ra+rf+lshp+recurrent': 
        ['009_ra+rf+lshp+recurrent-01', 
         '009_ra+rf+lshp+recurrent-02'],
    '010_ra+rf+lshp+recurrent+vec_obs': 
        ['010_ra+rf+lshp+recurrent+vec_obs-01'],
}


in_path = 'summaries'
out_path = 'charts'

for job_name in jobs.keys():
    paths = [os.path.join('.', in_path, n) for n in jobs[job_name]]
    d = tabulate_events(paths)
    write_combined_events(os.path.join('.', out_path, job_name), d)