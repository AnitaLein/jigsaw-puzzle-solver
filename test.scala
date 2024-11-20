import scala.io.StdIn.readLine

class Student(var name: String) {
  val github_username = name
}

@main
def main = {
  val students = Seq[Student](Student("tutor"))

  for(s<-students) println(s.github_username)
}

