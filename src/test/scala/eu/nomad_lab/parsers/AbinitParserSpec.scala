package eu.nomad_lab.parsers

import org.specs2.mutable.Specification

object AbinitParserSpec extends Specification {
  "AbinitParserTest" >> {
    "test Si with json-events" >> {
      ParserRun.parse(AbinitParser, "parsers/abinit/test/examples/Si/Si.out", "json-events") must_== ParseResult.ParseSuccess
    }
    "test Si with json" >> {
      ParserRun.parse(AbinitParser, "parsers/abinit/test/examples/Si/Si.out", "json") must_== ParseResult.ParseSuccess
    }
    "test Fe with json-events" >> {
      ParserRun.parse(AbinitParser, "parsers/abinit/test/examples/Fe/Fe.out", "json-events") must_== ParseResult.ParseSuccess
    }
    "test Fe with json" >> {
      ParserRun.parse(AbinitParser, "parsers/abinit/test/examples/Fe/Fe.out", "json") must_== ParseResult.ParseSuccess
    }
    "test H2 with json-events" >> {
      ParserRun.parse(AbinitParser, "parsers/abinit/test/examples/H2/H2.out", "json-events") must_== ParseResult.ParseSuccess
    }
    "test H2 with json" >> {
      ParserRun.parse(AbinitParser, "parsers/abinit/test/examples/H2/H2.out", "json") must_== ParseResult.ParseSuccess
    }
  }
}
