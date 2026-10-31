"""Synthetic-only parser tests: no private R0/GPU logs needed."""
import json
import unittest
from qpn_r0_aggregate import PREFIX, SCOPE, inspect_record, scrub_log


def record(**changes):
    x = dict(schema=2, scope=SCOPE, instrumented=True,
             coalesced_layer_envelopes=True, callback_accounting_complete=True,
             event_pairs=9, observed_layers=3, product_layers=2,
             h2d_operations=4, h2d_bytes=4*2764816,
             trace_window_ms=14.0, h2d_envelope_ms=8.0,
             products_envelope_ms=7.0, wait_envelope_ms=1.0,
             gather_envelope_ms=3.0, h2d_products_overlap_ms=2.0)
    x.update(changes)
    return x


class R0AggregateTests(unittest.TestCase):
    def test_aggregates_and_scope(self):
        raw = ('other private log line\n' + PREFIX + json.dumps(record()) + '\n' +
               PREFIX + json.dumps(record(event_pairs=15)) + '\n')
        report = scrub_log(raw)
        self.assertEqual(report['chunk_records'], 2)
        self.assertTrue(report['coalesced_layer_envelopes'])
        self.assertEqual(report['event_pairs'], 24)
        self.assertEqual(report['summed_per_chunk_envelopes_ms']['trace_window_ms'], 28.)
        self.assertIsNone(report['eligible_exposed_critical_path_ms'])
        self.assertEqual(report['r1_full_request_feasibility'], 'UNKNOWN')
        self.assertNotIn('private log line', json.dumps(report))
        self.assertNotIn('routes', json.dumps(report))

    def test_fail_closed_unknown_or_private_fields(self):
        for extra in ('routes', 'prompt', 'logits', 'layer', 'raw_intervals', 'model_path'):
            with self.subTest(field=extra), self.assertRaises(ValueError):
                inspect_record(record(**{extra: 'secret'}))
        with self.assertRaises(ValueError):
            scrub_log('strata unrelated: {}')

    def test_fail_closed_wrong_scope_and_invalid_events(self):
        for update in (dict(scope='full model speedup'), dict(instrumented=False),
                       dict(schema=1), dict(callback_accounting_complete=False),
                       dict(coalesced_layer_envelopes=False),
                       dict(event_pairs=241), dict(observed_layers=0),
                       dict(h2d_envelope_ms=float('nan')),
                       dict(trace_window_ms=1.0),
                       dict(h2d_products_overlap_ms=8.0),
                       dict(h2d_operations=-1), dict(observed_layers=49),
                       dict(product_layers=4), dict(h2d_operations=0),
                       dict(event_pairs=1)):
            with self.subTest(update=update), self.assertRaises(ValueError):
                inspect_record(record(**update))

    def test_large_callback_counts_under_small_pair_budget(self):
        # 15347 expert-copy calls existed in prior first-chunk MMQ evidence.
        # Coalescing prevents the previous 32768 pair ceiling from causing a
        # diagnostic failure when product/wait callbacks are also frequent.
        large = record(event_pairs=9, h2d_operations=15347,
                       h2d_bytes=15347 * 2764816)
        self.assertEqual(inspect_record(large)['h2d_operations'], 15347)

    def test_zero_event_record_and_boolean_rejected(self):
        empty = record(event_pairs=0, observed_layers=0, product_layers=0,
                       h2d_operations=0, h2d_bytes=0, trace_window_ms=0.,
                       h2d_envelope_ms=0., products_envelope_ms=0.,
                       wait_envelope_ms=0., gather_envelope_ms=0.,
                       h2d_products_overlap_ms=0.)
        self.assertEqual(inspect_record(empty)['event_pairs'], 0)
        with self.assertRaises(ValueError):
            inspect_record(record(h2d_bytes=True))


if __name__ == '__main__':
    unittest.main()
