# Коллекция стандартных аналитик

- `fire_detection.yaml` - детекция огня/пожара (ИНК) + скрипты (каски, номера цистерн, детекция огня, время работы станка)
- `rail_wagon_number.yaml` - распознавание номеров вагонов (КОКС)
- `CleantechPark.yaml` - CleantechPark (PPE (helmets, gloves, vests - чекает несколько секунд для аллерта на наличие; motion - fallen persons, safety zones - человек рядом с грузовым ТС)
- `NGU.yaml` - НГУ прокторинг
- `ГРЗ2.yaml` - распознавание номерных знаков автомобилей
- `Люди в области.yaml` - подсчет людей в кадре или в области
- `Маски_шапки.yaml` - 
- `Машины в зоне.yaml` - подсчет машин в зоне
- `Опасные зоны.yaml` - детецкия людей в опасной зоне
- `Парковка.yaml` - подсчет машин в зоне
- `Подсчёт людей.yaml` - сохранение в ивентах людей, пересекших одну из двух линий
- `Подсчёт машин.yaml` - охранение в ивентах машин, пересекших одну из двух линий
- `Сохранение лиц.yaml` - сохранение лиц в ивентах

- `vehicle_counter.yaml` - подсчет транспорта, въехавшего в зону; отдельные счетчики для каждого класса (автомобиль, грузовик, автобус, мотоцикл и т.д.)
- `car_entrance_check.yaml` - проверка наличия автомобиля и человека в зоне, при наличии объекта в зоне на кадре создается текст-бокс
- `car_idle.yaml` - проверка объекта на движение; если объект в зоне не двигается дольше заданного промежутка времени, то создается ивент
- `cashier_away.yaml` - проверка присутствия человека в зоне. Если в зоне нет человека дольше заданного промежутка времени, то создается ивент
- `empty_filling_station.yaml` - подсчет пустых мест на заправке
- `front_door_counter.yaml` - подсчет объектов, которые пересекли линию
- `person_in_danger_zone.yaml` - детекция объекта в опасной зоне с выводом соответствующего статуса фрейма