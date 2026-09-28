from datetime import date
def test_bar_series_accessors():
    from nasdaq_agent.sources.models import Bar, BarSeries
    bars = [Bar(date=date(2026, 9, 17), open=1, high=1, low=1, close=10.0, adj_close=9.5, volume=1),
            Bar(date=date(2026, 9, 18), open=1, high=1, low=1, close=10.2, adj_close=9.7, volume=1)]
    s = BarSeries(symbol="AAPL", bars=bars, source="test")
    assert s.closes() == [10.0, 10.2] and s.adj_closes() == [9.5, 9.7]
    assert s.dates() == [date(2026, 9, 17), date(2026, 9, 18)]


def test_most_recent_first_orders_by_publication_time_with_undated_headlines_last():
    from datetime import datetime, timezone
    from nasdaq_agent.sources.models import Headline, most_recent_first
    older = Headline(id=1, title="older", provider="p", published=datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc))
    undated = Headline(id=2, title="undated", provider="p")
    newer = Headline(id=3, title="newer", provider="p", published=datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc))
    naive_middle = Headline(id=4, title="naive, read as UTC", provider="p", published=datetime(2026, 9, 15, 12, 0))
    assert [h.id for h in most_recent_first([older, undated, newer, naive_middle])] == [3, 4, 1, 2]


def test_most_recent_first_keeps_the_given_order_for_ties():
    from datetime import datetime, timezone
    from nasdaq_agent.sources.models import Headline, most_recent_first
    t = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    same_time = [Headline(id=i, title=str(i), provider="p", published=t) for i in (2, 1, 3)]
    assert [h.id for h in most_recent_first(same_time)] == [2, 1, 3]
