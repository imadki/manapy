from termcolor import colored

class Entity:
  def __init__(self, name):
    self.name = name
    self.passed = 0
    self.failed = 0
    self.nbr_tests = 0
    self.error = None
    self.error_index = None

class TestLogger:

  def __init__(self):
    self.entities = {}
    self.sorted_entities = []

  def testing(self, label, assert_fun, *args, **keys):
    if label not in self.entities.keys():
      self.entities[label] = Entity(label)
      self.sorted_entities.append(label)
    self.entities[label].nbr_tests += 1
    try:
      assert_fun(*args, **keys)
      self.entities[label].passed += 1
    except AssertionError as e:
      self.entities[label].failed += 1
      if not self.entities[label].error:
        self.entities[label].error = str(e)
        self.entities[label].error_index = self.entities[label].nbr_tests
      return False
    return True


  def summary(self):
    failed_tests = 0
    passed_tests = 0
    for item in self.sorted_entities:
      if self.entities[item].error:
        failed_tests += 1
      else:
        passed_tests += 1
    number_of_tests = passed_tests + failed_tests

    print()
    print(colored("# ============================== #", "yellow"))
    print(colored("# ========== Summary =========== #", "yellow"))
    print(colored("# ============================== #", "yellow"))

    print(colored("===>", "blue"), end=' ')
    print(colored(f"{passed_tests} [P]", "green"), end=' ')
    print(colored(f"{failed_tests} [F]", "red"), end=' ')
    print(colored(f"Total {number_of_tests}", "blue"))
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print(colored(f"### Entities Details", "blue"))
    counter = 0
    for item in self.sorted_entities:
      counter += 1
      entity = self.entities[item]
      if entity.error:
        print(colored(f"* [F] {counter}/{number_of_tests}: {entity.name} {entity.error_index} / {entity.nbr_tests} {entity.failed} Failed", "red"))
        print(colored("\n".join("\t" + line for line in entity.error.splitlines()), "red"), end="")
        print()
      else:
        print(colored(f"[P] {counter}/{number_of_tests}: {entity.name}", "green"))

    print()
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    print(colored("===>", "blue"), end=' ')
    print(colored(f"{passed_tests} [P]", "green"), end=' ')
    print(colored(f"{failed_tests} [F]", "red"), end=' ')
    print(colored(f"Total {number_of_tests}", "blue"))

    return failed_tests == 0
