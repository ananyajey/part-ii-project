#vector = ['classical', 'rap', 'pop', 'hiphop', 'jazz'] #.....

vector = ['baroque', 'metal']

labels = {}

def get_uri(name):
    uri_dict = {'Bach, JS_ Brandenburg Concerto No. 2 in F Major, BWV 1047_ I. — _ Johann Sebastian Bach, Mark Bennet' : 'track:3AKAAbFYypDMas1MV8Nhot',
                'Bach, JS_ Brandenburg Concerto No. 5 in D Major, BWV 1050_ II. Affettuoso _ Johann Sebastian Bach, E' : '2O348yjmVxXiR8UkiBkZ1O',
                'Canon in D _ Johann Pachelbel, Kanon Orchestre de Chambre, Jean-François Paillard' : '1c3GkbZBnyrQ1cm4TGHFrK',
                'Messiah, HWV 56 _ Pt. 2_ _Hallelujah_ _ George Frideric Handel, Academy of St Martin in the Fields C' : '4TNCQyG2gmepsGoeLdRKn4',
                'Toccata and Fugue in D Minor, BWV 565_ I. Toccata _ Johann Sebastian Bach, Simon Preston' : '4HE2Ex0bjbj3YNXmV01OoM',
                'Vivaldi_ The Four Seasons, Violin Concerto in E Major, Op. 8 No. 1, RV 269 _Spring_ II. Largo e pian' : '7okHiZwiZD9nILstUvueqA',
                'Vivaldi_ The Four Seasons, Violin Concerto in F Minor, Op. 8 No. 4, RV 297 _Winter_ I. Allegro non m' : '0ON4FYmS4Zch1NV0lhv9hX'
    }
    return uri_dict[name]

def add_label(uri, genre):
    if genre == 'baroque':
        labels[uri] = [1,0]
    elif genre == 'metal':
        labels[uri] = [0,1]

def get_label(uri):
    return labels[uri]

def get_info(uri):
    # TODO: implement
    return

#add_label('012', 'baroque')
#print(labels)
