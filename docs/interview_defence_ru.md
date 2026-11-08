# Interview defence

**What did this project test?** “I implemented a controlled audit of small-model judgements using objectively labelled synthetic problems, counterbalanced answer order and paired style interventions.”

Развёрнуто: “The raw audit measures pairwise correctness, position sensitivity and rhetorical effects. A separate held-out experiment fits a single temperature on neutral calibration problems and tests Brier score and negative log-likelihood on disjoint problems.” Не утверждайте экспертное согласие: людей в разметке не было.

**Are these calibrated probabilities?** “They are conditional A/B probabilities or sigmoid reward margins. Temperature scaling is evaluated separately; neither probability construction guarantees calibration.” Строгий ответ должен опираться на фактический `temperature_calibration.json`, включая ухудшение, если оно есть.

**How did you avoid pseudoreplication?** “I treated the problem as the sampling unit, retaining answer-order repetitions within each cluster.” 640 решений не дают n=640 независимых задач.

**What did you personally implement?** Код подготовлен с помощью AI coding assistant и исполнен локально. До личного разбора и самостоятельного воспроизведения корректно говорить: “I developed and ran an AI-assisted research implementation, and can explain its protocol and limitations.” Не приписывайте себе ручное написание каждой строки. Для защиты самостоятельно воспроизведите один тест и объясните формулу Brier и зависимость результата от порядка.
