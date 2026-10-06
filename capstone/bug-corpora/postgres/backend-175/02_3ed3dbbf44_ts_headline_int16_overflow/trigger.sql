SELECT ts_headline('english', 'foo barbar', to_tsquery('english', 'foo'), 'StartSel=' || repeat('x', 32768));
