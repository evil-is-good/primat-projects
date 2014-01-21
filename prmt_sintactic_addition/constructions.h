
#ifndef PRMT_CONSTRUCTION

#define PRMT_CONSTRUCTION

// WARNING! Буть внимателен, индентификаторы, определённые в этом файле могут
// бать определены в подключаемых заголовках.

#define with_params /*!< `GridGenerator ::create_some_grid (goal, with_params 1, 2);` */
#define and_assigned_to /*!< `GridGenerator ::create_some_grid (1, 2, and_assigned_to doal);` */
#define assigned_to /*!< `GridGenerator ::create_some_grid (assigned_to goal);` */

#define _is == /*!< Чтоб писать `if (car _is vaz2101)`. */
#define then  /*!< `if (car _is vaz2101) then` :) */

#define FOR_I(begin, end) for (size_t i = begin; i < end; ++i) /*!< А это ещё более линивые форы. */
#define FOR_J(begin, end) for (size_t j = begin; j < end; ++j)
#define FOR_K(begin, end) for (size_t k = begin; k < end; ++k)
#define FOR_L(begin, end) for (size_t l = begin; l < end; ++l)
#define FOR_M(begin, end) for (size_t m = begin; m < end; ++m)
#define FOR_N(begin, end) for (size_t n = begin; n < end; ++n)
#define FOR_O(begin, end) for (size_t o = begin; o < end; ++o)
#define FOR_P(begin, end) for (size_t p = begin; p < end; ++p)
#define FOR(iter, begin, end) for(size_t iter = begin; iter < end; ++iter) /*!< 
Запись `for (size_t i = begin; i < end; ++i)` полюбас встречается заметно чаще остальных форов.
Так что, почему бы и не облегчит жизнь макросом `FOR (iter, begin, end)`? Можно, конечно юзать 
сниппеты. Ну собственно а почему бы и не так? В общем это спорный момент, не вижу опасности
или сложности доставляемых сим макросом, потому юзаю его.*/

#define LABEL(name) name: /*!< В тех редких лсучаях, когда нужня метка, записывать её более пристойно. */

#endif
