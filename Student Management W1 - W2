from flask import Flask, jsonify, request

app = Flask(__name__)

# Dummy Database
students = [
    {
        "id": 1,
        "name": "Anushka",
        "age": 20
    },
    {
        "id": 2,
        "name": "Arpeet",
        "age": 21
    }
]

# Home Route
@app.route("/")
def home():
    return "Student Management API"

# GET - Get All Students
@app.route("/students", methods=["GET"])
def get_students():
    return jsonify(students)

# GET - Get Single Student
@app.route("/students/<int:id>", methods=["GET"])
def get_student(id):
    for student in students:
        if student["id"] == id:
            return jsonify(student)

    return jsonify({"message": "Student Not Found"}), 404

# POST - Add Student
@app.route("/students", methods=["POST"])
def add_student():
    data = request.get_json()

    new_student = {
        "id": len(students) + 1,
        "name": data["name"],
        "age": data["age"]
    }

    students.append(new_student)

    return jsonify({
        "message": "Student Added Successfully",
        "student": new_student
    }), 201

# PUT - Update Student
@app.route("/students/<int:id>", methods=["PUT"])
def update_student(id):
    data = request.get_json()

    for student in students:
        if student["id"] == id:
            student["name"] = data["name"]
            student["age"] = data["age"]

            return jsonify({
                "message": "Student Updated Successfully",
                "student": student
            })

    return jsonify({"message": "Student Not Found"}), 404

# DELETE - Delete Student
@app.route("/students/<int:id>", methods=["DELETE"])
def delete_student(id):
    for student in students:
        if student["id"] == id:
            students.remove(student)

            return jsonify({
                "message": "Student Deleted Successfully"
            })

    return jsonify({"message": "Student Not Found"}), 404

# Run Server
if __name__ == "__main__":
    app.run(debug=True)
